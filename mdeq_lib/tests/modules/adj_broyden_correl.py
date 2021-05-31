import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import mdeq_lib.models as models
from mdeq_lib.config import config
from mdeq_lib.modules.adj_broyden import adj_broyden
from mdeq_lib.modules.broyden import broyden, rmatvec
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

def adj_broyden_correl(opa_freq, n_runs=1, random_prescribed=True, dataset='imagenet', model_size='LARGE'):
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
    methods_results = {
        method_name: {'correl': [], 'ratio': []}
        for method_name in ['shine-adj-br', 'shine', 'shine-opa', 'fpn']
    }
    methods_solvers = {
        'shine': broyden,
        'shine-adj-br': adj_broyden,
        'shine-opa': adj_broyden,
        'fpn': broyden,
    }
    random_results = {'correl': [], 'ratio': []}
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
        directions_dir = {
            'random': torch.randn(z1_est.shape),
            'prescribed': torch.randn(z1_est.shape),
        }
        for method_name in methods_results.keys():
            z1_est = torch.zeros_like(z1_est)
            g = lambda x: DEQFunc2d.g(model.fullstage_copy, x, x_list, cutoffs, *args)
            if random_prescribed:
                inverse_direction_fun = lambda x: directions_dir['prescribed']
            else:
                model.copy_modules()
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

            solver = methods_solvers[method_name]
            if 'opa' in method_name:
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
            Us = result_info['Us']
            VTs = result_info['VTs']
            nstep = result_info['lowest_step']
            if opa_freq is not None:
                nstep += (nstep-1)//opa_freq
            # compute true incoming gradient if needed
            if not random_prescribed:
                directions_dir['prescribed'] = inverse_direction_fun_vec(z1_est)
                # making sure the random direction norm is not unrealistic
                directions_dir['random'] = directions_dir['random'] * torch.norm(directions_dir['prescribed']) / torch.norm(directions_dir['random'])
            # inversion on random gradients
            z1_temp = z1_est.clone().detach().requires_grad_()
            with torch.enable_grad():
                y = DEQFunc2d.g(model.fullstage_copy, z1_temp, x_list, cutoffs, *args)

            eps = 2e-10
            for direction_name, direction in directions_dir.items():
                def g(x):
                    y.backward(x, retain_graph=True)
                    res = z1_temp.grad + direction
                    z1_temp.grad.zero_()
                    return res
                result_info_inversion = broyden(
                    g,
                    direction,  # we initialize Jacobian Free style
                    # in order to accelerate the convergence
                    threshold=35,
                    eps=eps,
                    name="backward",
                )
                true_inv = result_info_inversion['result']
                inv_dir = {
                    'fpn': direction,
                    'shine': - rmatvec(Us[:,:,:,:nstep-1], VTs[:,:nstep-1], direction),
                }
                inv_dir['shine-opa'] = inv_dir['shine']
                inv_dir['shine-adj-br'] = inv_dir['shine']
                approx_inv = inv_dir[method_name]
                correl = torch.dot(
                    torch.flatten(true_inv),
                    torch.flatten(approx_inv),
                )
                scaling = torch.norm(true_inv) * torch.norm(approx_inv)
                correl = correl / scaling
                ratio = torch.norm(true_inv) / torch.norm(approx_inv)
                if direction_name == 'prescribed':
                    methods_results[method_name]['correl'].append(correl.item())
                    methods_results[method_name]['ratio'].append(ratio.item())
                else:
                    if method_name == 'fpn':
                        random_results['correl'].append(correl.item())
                        random_results['ratio'].append(ratio.item())
            y.backward(torch.zeros_like(true_inv), retain_graph=False)
    return methods_results, random_results


def present_results(
        methods_results, random_results,
        opa_freq=None,
        random_prescribed=True,
        dataset='imagenet',
        model_size='SMALL',
):
    fig, axs = plt.subplots(
        1, 2, figsize=(5.5, 2.1),
        gridspec_kw=dict(width_ratios=[0.84, .15], wspace=.4),
    )
    naming = {
        'prescribed': 'Additional',
        'random': 'Random',
    }
    method_naming = {
        'shine': 'SHINE w. Broyden',
        'shine-adj-br': 'SHINE w. Adj. Broyden',
        'shine-opa': 'SHINE w. Adj. Broyden / OPA',
        'fpn': 'Jacobian-Free',
    }

    styles = {
        'shine': dict(color='C2', alpha=0.8),
        'fpn': dict(color='C1'),
        'shine-opa': dict(color='chocolate'),
        'shine-adj-br': dict(color='navajowhite', alpha=0.8),
    }
    ax_scatter = axs[0]
    method_names = 'fpn shine shine-adj-br shine-opa'.split()
    for method_name in method_names:
        method_results = methods_results[method_name]
        # ax_scatter.scatter(
        #     method_results['ratio'],
        #     method_results['correl'],
        #     # label=f"{method_naming[method_name]} - {np.median(method_results['correl']):.4f}",
        #     label=f"{method_naming[method_name]}",
        #     s=3.,
        #     **styles[method_name],
        # )
        sns.kdeplot(
            x=method_results['ratio'],
            y=method_results['correl'],
            ax=ax_scatter,
            label=f"{method_naming[method_name]}",
            cut=2,
            **styles[method_name],
        )
    # XXX: how can we include random inversion ?
    ax_scatter.set_ylabel(r'$\operatorname{cossim}(a, b)$')
    ax_scatter.set_xlabel(r'$\|a \|/\| b \|$')
    handles, labels = ax_scatter.get_legend_handles_labels()

    ### legend
    ax_legend = axs[-1]
    legend = ax_legend.legend(
        handles,
        labels,
        loc='center',
        ncol=1,
        handlelength=1.5,
        handletextpad=.1,
        # title=r'\textbf{Method} - median correlation',
        title=r'\textbf{Method}',
    )
    ax_legend.axis('off')
    fig_name = 'adj_broyden_inversion'
    if opa_freq is not None:
        fig_name += f'_opa{opa_freq}'
    if not random_prescribed:
        fig_name += '_true_grad'
    fig_name += f'_scatter_merged_{dataset}_{model_size}.pdf'
    plt.savefig(fig_name, dpi=300)


def save_results(
        n_runs=100,
        random_prescribed=False,
        dataset='imagenet',
        model_size='SMALL',
):
    methods_results, random_results = adj_broyden_correl(
        5,
        n_runs,
        random_prescribed,
        dataset,
        model_size,
    )
    res_name = f'adj_broyden_inv_results_merged_{dataset}_{model_size}'
    if not random_prescribed:
        res_name += '_true_grad'
    res_name += '.pkl'
    with open(res_name, 'wb') as f:
        pickle.dump((methods_results, random_results), f)


if __name__ == '__main__':
    random_prescribed = False
    opa_freq = 5
    dataset = 'cifar'
    model_size = 'LARGE'
    res_name = f'adj_broyden_inv_results_merged_{dataset}_{model_size}'
    if not random_prescribed:
        res_name += '_true_grad'
    res_name += '.pkl'
    with open(res_name, 'rb') as f:
        methods_results, random_results = pickle.load(f)
    present_results(
        methods_results, random_results,
        opa_freq=opa_freq,
        random_prescribed=random_prescribed,
        dataset=dataset,
        model_size=model_size,
    )
