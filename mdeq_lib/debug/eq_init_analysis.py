import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from mdeq_lib.config import config
from mdeq_lib.config.env_config import CHECKPOINTS_DIR
from mdeq_lib.core.cls_function import train
import mdeq_lib.models as models
from mdeq_lib.modules.optimizations import VariationalHidDropout2d
from mdeq_lib.training.cls_train import update_config_w_args
from mdeq_lib.utils.utils import get_optimizer



def set_dropout_modules_active(model):
    for m in model.modules():
        if isinstance(m, VariationalHidDropout2d):
            m.train()

def analyze_equilibrium_initialization(
    model_size='TINY',
    at_init=False,
    dataset='cifar',
    n_samples_train=1000,
    n_images=100,
    checkpoint=None,
    on_cpu=False,
    n_gpus=1,
    dropout_eval=False,
):
    update_config_w_args(
        n_epochs=100,
        pretrained=False,
        n_gpus=n_gpus,
        dataset=dataset,
        model_size=model_size,
        use_group_norm=False,
        n_refine=None,
    )
    model = models.mdeq.get_cls_net(config, shine=False, fpn=False, refine=False, fallback=False, adjoint_broyden=False)
    criterion = torch.nn.CrossEntropyLoss()
    if not on_cpu:
        torch.multiprocessing.set_start_method('spawn')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model = model.cuda()
        gpus = list(config.GPUS)
        model = nn.DataParallel(model, device_ids=gpus).cuda()
        criterion = criterion.cuda()
    checkpoint_name = 'checkpoint'
    if checkpoint is not None:
        checkpoint_name += f'_{checkpoint}'
    model_state_file = CHECKPOINTS_DIR / config.DATASET.DATASET / f'cls_mdeq_{model_size}_0/{checkpoint_name}.pth.tar'
    if not at_init:
        ckpt = torch.load(
            model_state_file,
            map_location=torch.device('cpu') if on_cpu else None,
        )
        if on_cpu:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.module.load_state_dict(ckpt['state_dict'])

    model.eval()
    if dropout_eval:
        set_dropout_modules_active(model)
    if dataset == 'cifar':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augment_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_valid)
        aug_train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
    else:
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augment_list = [
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
        ]
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_valid)
        aug_train_dataset = datasets.ImageFolder(traindir, transform_train)

    df_results = pd.DataFrame(columns=[
        'image_index',
        'before_training',
        'trace',
        'init_type',
        'is_aug',
        'i_iter',
    ])

    def fill_df_results(df_results, pickle_file_name,  **data_kwargs):
        result_info = pickle.load(open(f'{pickle_file_name}.pkl', 'rb'))
        trace = result_info['trace']
        i_iter = np.arange(len(trace))
        df_trace = pd.DataFrame(data={
            'trace' :trace,
            'i_iter': i_iter,
            **data_kwargs,
        })
        df_results = df_results.append(df_trace, ignore_index=True)
        return df_results

    image_indices = np.random.choice(len(train_dataset), n_images, replace=False)
    vanilla_inits = {}
    aug_inits = {}
    fn = model if on_cpu else model.module
    debug_info_tag = f'{dataset}_{model_size}_{checkpoint}_{n_samples_train}'
    for image_index in image_indices:
        image, _ = train_dataset[image_index]
        image = image.unsqueeze(0)
        if not on_cpu:
            image = image.cuda()
        # pot in kwargs we can have: f_thres, b_thres, lim_mem
        _, y_list, _ = fn(image, train_step=-1, index=image_index, debug_info=f'before_training_{debug_info_tag}')
        vanilla_inits[image_index] = y_list
        df_results = fill_df_results(
            df_results,
            f'result_info_before_training_{debug_info_tag}',
            image_index=image_index,
            before_training=True,
            init_type=None,
            is_aug=False,
        )
        aug_image, _ = aug_train_dataset[image_index]
        aug_image = aug_image.unsqueeze(0)
        if not on_cpu:
            aug_image = aug_image.cuda()
        _, aug_y_list, _ = fn(aug_image, train_step=-1, index=image_index)
        aug_inits[image_index] = aug_y_list

    aug_train_loader = torch.utils.data.DataLoader(
        Subset(aug_train_dataset, list(range(n_samples_train))),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus) if not on_cpu else config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    optimizer = get_optimizer(config, model)
    if not at_init:
        optimizer.load_state_dict(ckpt['optimizer'])
    last_epoch = 0
    if not at_init:
        last_epoch = ckpt['lr_scheduler']['last_epoch']
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e5,
                        last_epoch=last_epoch)
    if not at_init:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        last_epoch = ckpt['epoch']
    train(
        config,
        aug_train_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        last_epoch,
        None,
        None,
        None,
        global_steps=config.TRAIN.PRETRAIN_STEPS+1,
    )
    model.eval()
    if dropout_eval:
        set_dropout_modules_active(model)
    for image_index in image_indices:
        image, _ = train_dataset[image_index]
        image = image.unsqueeze(0)
        if not on_cpu:
            image = image.cuda()
        _ = fn(image, train_step=-1, index=image_index, debug_info=f'after_training_{debug_info_tag}')
        df_results = fill_df_results(
            df_results,
            f'result_info_after_training_{debug_info_tag}',
            image_index=image_index,
            before_training=False,
            init_type=None,
            is_aug=False,
        )
        _ = fn(image, train_step=-1, index=image_index, debug_info=f'after_training_init_{debug_info_tag}', z_0=vanilla_inits[image_index])
        df_results = fill_df_results(
            df_results,
            f'result_info_after_training_init_{debug_info_tag}',
            image_index=image_index,
            before_training=False,
            init_type='vanilla',
            is_aug=False,
        )
        new_aug_image, _ = aug_train_dataset[image_index]
        new_aug_image = new_aug_image.unsqueeze(0)
        if not on_cpu:
            new_aug_image = new_aug_image.cuda()
        _ = fn(new_aug_image, train_step=-1, index=image_index, debug_info=f'after_training_init_aug_aug_{debug_info_tag}', z_0=aug_inits[image_index])
        df_results = fill_df_results(
            df_results,
            f'result_info_after_training_init_aug_aug_{debug_info_tag}',
            image_index=image_index,
            before_training=False,
            init_type='aug',
            is_aug=True,
        )
    results_name = f'eq_init_results_{dataset}_{model_size}_{n_samples_train}'
    if checkpoint:
        results_name += f'_ckpt{checkpoint}'
    if dropout_eval:
        results_name += '_dropout'
    df_results.to_csv(
        f'{results_name}.csv',
    )
    return df_results
