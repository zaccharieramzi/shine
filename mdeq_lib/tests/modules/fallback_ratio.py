import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import mdeq_lib.models as models
from mdeq_lib.config import config
from mdeq_lib.modules.broyden import broyden, rmatvec
from mdeq_lib.modules.deq2d import DEQFunc2d
from mdeq_lib.training.cls_train import update_config_w_args, worker_init_fn, partial
from mdeq_lib.utils.utils import create_logger


def setup_model(opa=False, dataset='imagenet', model_size='SMALL'):
    seed = 42
    restart_from = 50
    n_epochs = 100
    pretrained = False
    n_gpus = 1
    use_group_norm = False
    shine = False
    fpn = False
    adjoint_broyden = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
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

def fallback_ratio(n_runs=1, dataset='imagenet', model_size='SMALL'):
    # setup
    model = setup_model(False, dataset, model_size)
    if dataset == 'imagenet':
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
        num_workers=10,
        pin_memory=True,
        worker_init_fn=partial(worker_init_fn, seed=42),
    )
    fallback_uses = 0
    iter_loader = iter(train_loader)
    for i_run in range(n_runs):
        input, target = next(iter_loader)
        target = target.cuda(non_blocking=True)
        x_list, z_list = model.feature_extraction(input.cuda())
        model.fullstage._reset(z_list)
        model.fullstage_copy._copy(model.fullstage)
        # fixed point solving
        x_list = [x.clone().detach().requires_grad_() for x in x_list]
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]
        args = (27, int(1e9), None)
        nelem = sum([elem.nelement() for elem in z_list])
        eps = 1e-5 * np.sqrt(nelem)
        z1_est = DEQFunc2d.list2vec(z_list)
        z1_est = torch.zeros_like(z1_est)
        g = lambda x: DEQFunc2d.g(model.fullstage_copy, x, x_list, cutoffs, *args)
        model.copy_modules()
        loss_function = lambda y_est: model.get_fixed_point_loss(y_est, target)
        def inverse_direction_fun(x):
            x_temp = x.clone().detach().requires_grad_()
            with torch.enable_grad():
                x_list = DEQFunc2d.vec2list(x_temp, cutoffs)
                loss = loss_function(x_list)
            loss.backward()
            dl_dx = x_temp.grad
            return dl_dx

        result_info = broyden(
            g,
            z1_est,
            threshold=config.MODEL.F_THRES,
            eps=eps,
            name="forward",
        )
        z1_est = result_info['result']
        Us = result_info['Us']
        VTs = result_info['VTs']
        nstep = result_info['lowest_step']
        # compute true incoming gradient
        grad = inverse_direction_fun(z1_est)

        inv_dir =  - rmatvec(Us[:,:,:,:nstep-1], VTs[:,:nstep-1], grad)
        fallback_mask = inv_dir.view(32, -1).norm(dim=1) > 1.8 * grad.view(32, -1).norm(dim=1)
        fallback_uses += fallback_mask.sum().item()
    return fallback_uses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get fallback use with SHINE.')
    parser.add_argument('--dataset', '-d', default='imagenet',
                        help='The dataset to chose between cifar and imagenet.'
                        'Defaults to imagenet.')
    parser.add_argument('--n_runs', '-n', default=100,
                        help='Number of seeds to use for the figure. Defaults to 100.')
    args = parser.parse_args()
    dataset = args.dataset
    model_size = 'LARGE' if dataset == 'cifar' else 'SMALL'
    n_fallbacks = fallback_ratio(n_runs=int(args.n_runs), dataset=args.dataset, model_size=model_size)
    print(f'Fallback was used {n_fallbacks} times, a ratio of {n_fallbacks/32}')
