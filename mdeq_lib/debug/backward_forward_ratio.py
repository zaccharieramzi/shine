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
from mdeq_lib.modules.deq2d import DEQFunc2d
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

def eval_ratio_fb_classifier(
    n_epochs=100,
    pretrained=False,
    n_gpus=1,
    dataset='imagenet',
    model_size='SMALL',
    shine=False,
    fpn=False,
    gradient_correl=False,
    gradient_ratio=False,
    adjoint_broyden=False,
    opa=False,
    refine=False,
    fallback=False,
    save_at=None,
    restart_from=None,
    use_group_norm=False,
    seed=0,
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
    )
    print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logger, final_output_dir, tb_log_dir = create_logger(
        config,
        args.cfg,
        'train',
        shine=shine,
        fpn=fpn,
        seed=seed,
        use_group_norm=use_group_norm,
        adjoint_broyden=adjoint_broyden,
        opa=opa,
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
        adjoint_broyden=adjoint_broyden,
        refine=refine,
        fallback=fallback,
        opa=opa,
    ).cuda()

    # dump_input = torch.rand(config.TRAIN.BATCH_SIZE_PER_GPU, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]).cuda()
    # logger.info(get_model_summary(model, dump_input))

    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        logger.info(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))

    # copy model file
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)

    gpus = list(config.GPUS)
    print("Finished constructing model!")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


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

    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
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
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        worker_init_fn=partial(worker_init_fn, seed=seed),
    )


    iter_loader = iter(train_loader)
    ratios = []
    with torch.autograd.profiler.profile(use_cuda=True, with_stack=True) as prof:
        for i_sample in range(n_samples):
            input, target = next(iter_loader)
            input = input.cuda(non_blocking=False)
            x_list, z_list = model.feature_extraction(input)
            # For variational dropout mask resetting and weight normalization re-computations
            model.fullstage._reset(z_list)
            model.fullstage_copy._copy(model.fullstage)
            x_list = [x.clone().detach().requires_grad_() for x in x_list]
            z_list = [z.clone().detach().requires_grad_() for z in z_list]
            start_forward = time.time()
            with torch.no_grad():
                z_list = model.fullstage_copy(z_list, x_list)
                torch.cuda.synchronize()
            end_forward = time.time()
            time_forward = end_forward - start_forward
            z_list = model.fullstage_copy(z_list, x_list)
            z = DEQFunc2d.list2vec(z_list)
            start_backward = time.time()
            z.backward(z)
            torch.cuda.synchronize()
            end_backward = time.time()
            time_backward = end_backward - start_backward
            ratios.append(time_backward / time_forward)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    return ratios
