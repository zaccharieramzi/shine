import os
import random

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import mdeq_lib.models as models
from mdeq_lib.config import config
from mdeq_lib.training.cls_train import update_config_w_args
from mdeq_lib.utils.utils import create_logger


def setup_model(opa=False):
    seed = 42
    restart_from = 50
    n_epochs = 100
    pretrained = False
    n_gpus = 1
    dataset = 'imagenet'
    model_size = 'SMALL'
    use_group_norm = False
    shine = False
    fpn = False
    adjoint_broyden = True
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
    )

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config,
        shine=shine,
        fpn=fpn,
        gradient_correl=False,
        gradient_ratio=False,
        adjoint_broyden=adjoint_broyden,
        opa=opa,
    ).cuda()


    resume_file = f'checkpoint_{restart_from}.pth.tar'
    model_state_file = os.path.join(final_output_dir, resume_file)
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model

@pytest.mark.parametrize('opa', [True, False])
def test_adj_broyden(opa):
    model = setup_model(opa)
    traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(traindir, transform_train)
    input, target = train_dataset[0]
    model(input[None].cuda(), train_step=int(1e9), y=torch.tensor(target)[None].cuda())
