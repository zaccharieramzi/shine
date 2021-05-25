import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import mdeq_lib.models as models
from mdeq_lib.config import config
from mdeq_lib.modules.adj_broyden import adj_broyden
from mdeq_lib.modules.broyden import broyden
from mdeq_lib.modules.deq2d import DEQFunc2d
from mdeq_lib.training.cls_train import update_config_w_args, worker_init_fn, partial
from mdeq_lib.utils.utils import create_logger


plt.style.use(['science'])
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

def setup_model(opa=False, dataset='imagenet', model_size='SMALL'):
    seed = 42
    restart_from = 50
    n_epochs = 100
    pretrained = False
    n_gpus = 1
    use_group_norm = False
    shine = False
    fpn = False
    adjoint_broyden = True
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

def adj_broyden_convergence(opa_freq, n_runs=1, dataset='imagenet', model_size='LARGE'):
    # setup
    model = setup_model(opa_freq is not None, dataset, model_size)
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
    iter_loader = iter(train_loader)
    convergence_results = {
        'correl': [],
        'ratio': [],
        'diff': [],
        'rdiff': [],
    }
    for i_run in range(n_runs):
        input, target = next(iter_loader)
        target = target.cuda(non_blocking=True)
        solvers = {
            'adj_broyden': adj_broyden,
            'broyden': broyden,
        }
        solvers_results = {}
        for solver_name, solver in solvers.items():
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
            g = lambda x: DEQFunc2d.g(model.fullstage_copy, x, x_list, cutoffs, *args)
            model.copy_modules()
            if solver_name == 'adj_broyden':
                loss_function = lambda y_est: model.get_fixed_point_loss(y_est, target)
                def inverse_direction_fun_vec(x):
                    x_temp = x.clone().detach().requires_grad_()
                    with torch.enable_grad():
                        x_list = DEQFunc2d.vec2list(x_temp, cutoffs)
                        loss = loss_function(x_list)
                    loss.backward()
                    dl_dx = x_temp.grad
                    return dl_dx
                inverse_direction_fun = inverse_direction_fun_vec
                add_kwargs = dict(
                    inverse_direction_freq=opa_freq,
                    inverse_direction_fun=inverse_direction_fun if opa_freq is not None else None,
                )
            else:
                add_kwargs = {}
            result_info = solver(
                g,
                z1_est,
                threshold=config.MODEL.F_THRES,
                eps=eps,
                name="forward",
                **add_kwargs,
            )
            z1_est = result_info['result']
            solvers_results[solver_name] = z1_est.clone().detach()
        z1_adj_br = solvers_results['adj_broyden']
        z1_br = solvers_results['broyden']
        correl = torch.dot(
            torch.flatten(z1_adj_br),
            torch.flatten(z1_br),
        )
        scaling = torch.norm(z1_adj_br) * torch.norm(z1_br)
        convergence_results['correl'].append((correl / scaling).item())
        convergence_results['ratio'].append((torch.norm(z1_br) / torch.norm(z1_adj_br)).item())
        convergence_results['diff'].append(torch.norm(z1_br - z1_adj_br).item())
        convergence_results['rdiff'].append((torch.norm(z1_br - z1_adj_br) / torch.norm(z1_br)).item())

    return convergence_results


def present_results(
        conv_results,
):
    for metric_name, metric_values in conv_results.items():
        print(metric_name, np.mean(metric_values), np.std(metric_values))


def save_results(
        n_runs=100,
        dataset='imagenet',
        model_size='SMALL',
):
    for opa_freq in [None, 1, 5]:
        print('='*20)
        if opa_freq is not None:
            print(f'With OPA {opa_freq}')
        else:
            print('Without OPA')
        convergence_results = adj_broyden_convergence(
            opa_freq,
            n_runs,
            dataset,
            model_size,
        )
        res_name = f'adj_broyden_conv_results_{dataset}_{model_size}'
        if opa_freq is not None:
            res_name += f'_opa{opa_freq}'
        res_name += '.pkl'
        with open(res_name, 'wb') as f:
            pickle.dump(convergence_results, f)


if __name__ == '__main__':
    n_runs = 100
    random_prescribed = False
    save_results = False
    reload_results = False
    plot_results = True
    dataset = 'cifar'
    model_size = 'LARGE'
    print('Ratio is true inv over approx inv')
    print('Results are presented: method, median correl, median ratio')
    for opa_freq in [None, 1, 5]:
        print('='*20)
        if opa_freq is not None:
            print(f'With OPA {opa_freq}')
        else:
            print('Without OPA')
        res_name = f'adj_broyden_conv_results_{dataset}_{model_size}'
        if opa_freq is not None:
            res_name += f'_opa{opa_freq}'
        res_name += '.pkl'
        with open(res_name, 'rb') as f:
            inv_quality_results = pickle.load(f)
        present_results(
            inv_quality_results,
        )
        print('='*20)
