# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import namedtuple
from pathlib import Path
import pprint
import shutil
import sys

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

def update_config_w_args(n_epochs=100, pretrained=False, n_gpus=1, dataset='imagenet', model_size='SMALL'):
    if dataset == 'imagenet':
        data_dir = IMAGENET_DIR
    else:
        data_dir = CIFAR_DIR
    opts = [
        'DATASET.ROOT', str(data_dir) + '/',
        'GPUS', list(range(n_gpus)),
        'TRAIN.END_EPOCH', n_epochs,
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

def train_classifier(
    n_epochs=100,
    pretrained=False,
    n_gpus=1,
    dataset='imagenet',
    model_size='SMALL',
    shine=False,
    fpn=False,
    save_at=None,
    seed=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    args = update_config_w_args(
        n_epochs=n_epochs,
        pretrained=pretrained,
        n_gpus=n_gpus,
        dataset=dataset,
        model_size=model_size,
    )
    print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train', shine=shine, fpn=fpn)

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
    ).cuda()

    dump_input = torch.rand(config.TRAIN.BATCH_SIZE_PER_GPU, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]).cuda()
    logger.info(get_model_summary(model, dump_input))

    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        logger.info(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    print("Finished constructing model!")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)
    lr_scheduler = None

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])

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
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        valid_dataset = datasets.ImageFolder(valdir, transform_valid)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 is supported at this phase"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if config.DATASET.AUGMENT else []
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
        valid_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=False, download=True, transform=transform_valid)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
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

    # Training code
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        topk = (1,5) if dataset_name == 'imagenet' else (1,)
        if config.TRAIN.LR_SCHEDULER == 'step':
            lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
              final_output_dir, tb_log_dir, writer_dict, topk=topk)
        torch.cuda.empty_cache()

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion, lr_scheduler, epoch,
                                  final_output_dir, tb_log_dir, writer_dict, topk=topk)
        torch.cuda.empty_cache()
        writer_dict['writer'].flush()

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        checkpoint_files = ['checkpoint.pth.tar']
        if save_at is not None and save_at == epoch:
            checkpoint_files.append(f'checkpoint_{epoch}.pth.tar')
        for checkpoint_file in checkpoint_files:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': config.MODEL.NAME,
                'state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, best_model, final_output_dir, filename=checkpoint_file)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
