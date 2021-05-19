import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import mdeq_lib.models as models
from mdeq_lib.config import config
from mdeq_lib.modules.adj_broyden import adj_broyden
from mdeq_lib.modules.broyden import broyden, rmatvec
from mdeq_lib.modules.deq2d import DEQFunc2d
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

def adj_broyden_correl(opa, n_runs=1):
    # setup
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
    x_list, z_list = model.feature_extraction(input[None].cuda())
    # fixed point solving
    x_list = [x.clone().detach() for x in x_list]
    cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]
    args = (27, int(1e9), None)
    nelem = sum([elem.nelement() for elem in z_list])
    eps = 1e-5 * np.sqrt(nelem)
    z1_est = DEQFunc2d.list2vec(z_list)
    inv_quality_results = {dir: {
        k: {'correl': [], 'ratio': []} for k in ['shine', 'fpn']
    } for dir in ['prescribed', 'random']}
    for i_run in range(n_runs):
        g = lambda x: DEQFunc2d.g(model.fullstage_copy, x, x_list, cutoffs, *args)
        directions_dir = {
            'random': torch.randn(z1_est.shape),
            'prescribed': torch.randn(z1_est.shape),
        }
        result_info = adj_broyden(
            g,
            z1_est,
            threshold=27,
            eps=eps,
            name="forward",
            inverse_direction_freq=3,
            inverse_direction_fun=lambda x: directions_dir['prescribed'],
        )
        z1_est = result_info['result']
        Us = result_info['Us']
        VTs = result_info['VTs']
        nstep = result_info['lowest_step']
        # inversion on random gradients

        z1_temp = z1_est.clone().detach().requires_grad_()
        with torch.enable_grad():
            y = DEQFunc2d.g(model.fullstage_copy, z1_temp, x_list, cutoffs, *args)

        eps = 2e-10
        for direction in inv_quality_results.keys():
            def g(x):
                y.backward(x, retain_graph=True)
                res = z1_temp.grad + directions_dir[direction]
                z1_temp.grad.zero_()
                return res
            result_info_inversion = broyden(
                g,
                directions_dir[direction],  # we initialize Jacobian Free style
                # in order to accelerate the convergence
                threshold=35,
                eps=eps,
                name="backward",
            )
            true_inv = result_info_inversion['result']
            inv_dir = {
                'fpn': directions_dir[direction],
                'shine': - rmatvec(Us[:,:,:,:nstep-1], VTs[:,:nstep-1], directions_dir[direction]),
            }
            for method in inv_quality_results[direction].keys():
                approx_inv = inv_dir[method]
                correl = torch.dot(
                    torch.flatten(true_inv),
                    torch.flatten(approx_inv),
                )
                scaling = torch.norm(true_inv) * torch.norm(approx_inv)
                correl = correl / scaling
                ratio = torch.norm(true_inv) / torch.norm(approx_inv)
                inv_quality_results[direction][method]['correl'].append(correl.item())
                inv_quality_results[direction][method]['ratio'].append(ratio.item())
        y.backward(torch.zeros_like(true_inv), retain_graph=False)
    return inv_quality_results


if __name__ == '__main__':
    adj_broyden_correl(False)
    adj_broyden_correl(True)
