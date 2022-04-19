import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from mdeq_lib.config import config
from mdeq_lib.core.cls_function import train, validate
import mdeq_lib.models as models
from mdeq_lib.training.cls_train import update_config_w_args
from mdeq_lib.utils.utils import get_optimizer



def analyze_equilibrium_initialization(
    model_size='TINY',
    at_init=False,
    dataset='cifar',
    n_samples_train=1000,
    n_images=100,
):

    args = update_config_w_args(
        n_epochs=100,
        pretrained=False,
        n_gpus=0,
        dataset='cifar',
        model_size='TINY',
        use_group_norm=False,
        n_refine=None,
    )
    model = models.mdeq.get_cls_net(config, shine=False, fpn=False, refine=False, fallback=False, adjoint_broyden=False)
    model_state_file = 'cls_mdeq_TINY_0/checkpoint.pth.tar'
    checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
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
    for image_index in image_indices:
        image, _ = train_dataset[image_index]
        image = image.unsqueeze(0)
        # pot in kwargs we can have: f_thres, b_thres, lim_mem
        _, y_list = model(image, train_step=-1, index=image_index, debug_info='before_training')
        vanilla_inits[image_index] = y_list
        df_results = fill_df_results(
            df_results,
            'result_info_before_training',
            image_index=image_index,
            before_training=True,
            init_type=None,
            is_aug=False,
        )
        aug_image, _ = aug_train_dataset[image_index]
        aug_image = aug_image.unsqueeze(0)
        _, aug_y_list = model(aug_image, train_step=-1, index=image_index)
        aug_inits[image_index] = aug_y_list

    aug_train_loader = torch.utils.data.DataLoader(
        Subset(aug_train_dataset, list(range(1000))),
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    optimizer = get_optimizer(config, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e5,
                        last_epoch=checkpoint['lr_scheduler']['last_epoch'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    last_epoch = checkpoint['epoch']
    train(
        config,
        aug_train_loader,
        model,
        torch.nn.CrossEntropyLoss(),
        optimizer,
        lr_scheduler,
        last_epoch,
        None,
        None,
        None,
    )
    model.eval()
    for image_index in image_indices:
        image, _ = train_dataset[image_index]
        _ = model(image, train_step=-1, index=image_index, debug_info='after_training')
        df_results = fill_df_results(
            df_results,
            'result_info_after_training',
            image_index=image_index,
            before_training=False,
            init_type=None,
            is_aug=False,
        )
        _ = model(image, train_step=-1, index=image_index, debug_info='after_training_init', z_0=vanilla_inits[image_index])
        df_results = fill_df_results(
            df_results,
            'result_info_after_training_init',
            image_index=image_index,
            before_training=False,
            init_type='vanilla',
            is_aug=False,
        )
        new_aug_image, _ = aug_train_dataset[image_index]
        new_aug_image = new_aug_image.unsqueeze(0)
        _ = model(new_aug_image, train_step=-1, index=image_index, debug_info='after_training_init_aug_aug', z_0=aug_inits[image_index])
        df_results = fill_df_results(
            df_results,
            'result_info_after_training_init_aug_aug',
            image_index=image_index,
            before_training=False,
            init_type='aug',
            is_aug=True,
        )
    df_results.to_csv(
        'eq_init_results_{dataset}_{model_size}_{n_samples_train}.csv',
    )
    return df_results
