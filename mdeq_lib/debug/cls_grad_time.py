# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from functools import partial
from collections import namedtuple
from pathlib import Path
import pprint
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import mdeq_lib.models as models
from mdeq_lib.config import config
from mdeq_lib.config import update_config
from mdeq_lib.config.env_config import (
    LOGS_DIR,
    CHECKPOINTS_DIR,
    DATA_DIR,
    CONFIG_DIR,
    IMAGENET_DIR,
    CIFAR_DIR,
    WORK_DIR,
)
from mdeq_lib.core.cls_function import train, validate
from mdeq_lib.utils.modelsummary import get_model_summary
from mdeq_lib.utils.utils import get_optimizer
from mdeq_lib.utils.utils import save_checkpoint
from mdeq_lib.utils.utils import create_logger
from termcolor import colored


# Set of argument to pass to different functions
Args = namedtuple(
    'Args',
    'cfg logDir modelDir dataDir testModel percent local_rank opts'.split()
)

def worker_init_fn(worker_id, seed=0):
    """Helper to make random number generation independent in each process."""
    np.random.seed(8*seed + worker_id)

def update_config_w_args(
    n_epochs=100,
    pretrained=False,
    n_gpus=1,
    dataset='imagenet',
    model_size='SMALL',
    use_group_norm=False,
    n_refine=None,
):
    if dataset == 'imagenet':
        data_dir = IMAGENET_DIR
    else:
        data_dir = CIFAR_DIR
    opts = [
        'DATASET.ROOT', str(data_dir) + '/',
        'GPUS', list(range(n_gpus)),
        'TRAIN.END_EPOCH', n_epochs,
        'MODEL.EXTRA.FULL_STAGE.GROUP_NORM', use_group_norm,
    ]
    if pretrained:
        opts += [
            'MODEL.PRETRAINED', str(WORK_DIR / 'pretrained_models' / f'MDEQ_{model_size}_Cls.pkl'),
            'TRAIN.PRETRAIN_STEPS', 0,
        ]
    if n_refine is not None:
        opts += [
            'MODEL.B_THRES', n_refine,
        ]
    args = Args(
        cfg=str(CONFIG_DIR / dataset / f'cls_mdeq_{model_size}.yaml'),
        logDir=str(LOGS_DIR) + '/',
        modelDir=str(CHECKPOINTS_DIR) + '/',
        dataDir=str(data_dir) + '/',
        testModel='',
        percent=1.0,
        local_rank=0,
        opts=opts,
    )
    update_config(config, args)
    return args

def train_classifier(
    n_epochs=100,
    pretrained=False,
    n_gpus=1,
    dataset='imagenet',
    model_size='SMALL',
    shine=False,
    fpn=False,
    fallback=False,
    refine=False,
    n_refine=None,
    gradient_correl=False,
    gradient_ratio=False,
    save_at=None,
    restart_from=None,
    use_group_norm=False,
    seed=0,
    compute_partial=True,
    compute_total=True,
    f_thres_range=range(2, 200),
    n_samples=1,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args = update_config_w_args(
        n_epochs=n_epochs,
        pretrained=pretrained,
        n_gpus=n_gpus,
        dataset=dataset,
        model_size=model_size,
        use_group_norm=use_group_norm,
        n_refine=n_refine,
    )
    print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logger, final_output_dir, tb_log_dir = create_logger(
        config,
        args.cfg,
        'train',
        shine=shine,
        fpn=fpn,
        seed=seed,
        use_group_norm=use_group_norm,
        refine=refine,
        fallback=fallback,
    )

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config,
        shine=shine,
        fpn=fpn,
        gradient_correl=gradient_correl,
        gradient_ratio=gradient_ratio,
        refine=refine,
        fallback=fallback,
    ).cuda()

    dump_input = torch.rand(config.TRAIN.BATCH_SIZE_PER_GPU, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]).cuda()
    logger.info(get_model_summary(model, dump_input))

    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        logger.info(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))

    # copy model file
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)

    writer_dict = None

    print("Finished constructing model!")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)
    lr_scheduler = None

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        if restart_from is None:
            resume_file = 'checkpoint.pth.tar'
        else:
            resume_file = f'checkpoint_{restart_from}.pth.tar'
        model_state_file = os.path.join(final_output_dir, resume_file)
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])

            # Update weight decay if needed
            checkpoint['optimizer']['param_groups'][0]['weight_decay'] = config.TRAIN.WD
            optimizer.load_state_dict(checkpoint['optimizer'])

            if 'lr_scheduler' in checkpoint:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e5,
                                  last_epoch=checkpoint['lr_scheduler']['last_epoch'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            best_model = True

    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
        valdir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TEST_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 is supported at this phase"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if config.DATASET.AUGMENT else []
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=partial(worker_init_fn, seed=seed),
    )

    # Learning rate scheduler
    if lr_scheduler is None:
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(train_loader)*config.TRAIN.END_EPOCH, eta_min=1e-6)
        elif isinstance(config.TRAIN.LR_STEP, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
    # train for one epoch
    forward_accelerated_times = []
    backward_accelerated_times = []
    forward_original_times = []
    backward_original_times = []
    data_iter = iter(train_loader)
    for i_data in range(n_samples):
        input, target = next(data_iter)
        model.train()
        if compute_partial:
            model.deq.shine = shine
            model.deq.fpn = fpn
            model.deq.gradient_ratio = gradient_ratio
            model.deq.gradient_correl = gradient_correl
            model.deq.refine = refine
            model.deq.fallback = fallback
            for f_thres in f_thres_range:
                model.f_thres = f_thres
                start_forward = time.time()
                output = model(input.cuda(), train_step=-(i_data+1), writer=None)
                end_forward = time.time()
                forward_accelerated_times.append(end_forward - start_forward)
                start_backward = time.time()
                target = target.cuda(non_blocking=True)

                loss = criterion(output, target)

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                end_backward = time.time()
                backward_accelerated_times.append(end_backward - start_backward)

        if compute_total:
            # model.f_thres = 30
            model.deq.shine = False
            model.deq.fpn = False
            model.deq.gradient_ratio = False
            model.deq.gradient_correl = False
            model.deq.refine = False
            model.deq.fallback = False
            for f_thres in f_thres_range:
                model.f_thres = f_thres
                start_forward = time.time()
                output = model(input.cuda(), train_step=-(i_data+1), writer=None)
                end_forward = time.time()
                forward_original_times.append(end_forward - start_forward)
                start_backward = time.time()
                target = target.cuda(non_blocking=True)

                loss = criterion(output, target)

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                end_backward = time.time()
                backward_original_times.append(end_backward - start_backward)
    method_name = Path(final_output_dir).name
    torch.save(torch.tensor(forward_accelerated_times), f'{method_name}_forward_times.pt')
    torch.save(torch.tensor(backward_accelerated_times), f'{method_name}_backward_times.pt')
    torch.save(torch.tensor(forward_original_times), f'{model_size}_original_forward_times.pt')
    torch.save(torch.tensor(backward_original_times), f'{model_size}_original_backward_times.pt')
