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

def adj_broyden_correl(opa, n_runs=1, random_prescribed=True, dataset='imagenet', model_size='LARGE'):
    # setup
    model = setup_model(opa, dataset, model_size)
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
    inv_quality_results = {dir: {
        k: {'correl': [], 'ratio': []} for k in ['shine', 'fpn']
    } for dir in ['prescribed', 'random']}
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
        g = lambda x: DEQFunc2d.g(model.fullstage_copy, x, x_list, cutoffs, *args)
        directions_dir = {
            'random': torch.randn(z1_est.shape),
            'prescribed': torch.randn(z1_est.shape),
        }
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
        result_info = adj_broyden(
            g,
            z1_est,
            threshold=config.MODEL.F_THRES,
            eps=eps,
            name="forward",
            inverse_direction_freq=1 if opa else None,
            inverse_direction_fun=inverse_direction_fun if opa else None,
        )
        z1_est = result_info['result']
        Us = result_info['Us']
        VTs = result_info['VTs']
        nstep = result_info['lowest_step']
        # compute true incoming gradient if needed
        if not random_prescribed:
            directions_dir['prescribed'] = inverse_direction_fun_vec(z1_est)
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


def present_results(
        inv_quality_results,
        opa=False,
        random_prescribed=True,
        dataset='imagenet',
        model_size='SMALL',
):
    fig = plt.figure(figsize=(5.5, 2.1))
    g = plt.GridSpec(1, 3, width_ratios=[0.42, 0.42, .15], wspace=.3)
    for i in range(3):
        ax = fig.add_subplot(g[0, i])
    allaxes = fig.get_axes()
    styles = {
        'prescribed': dict(color='C2', marker='o'),
        'random': dict(color='C0', marker='^'),
    }
    naming = {
        'prescribed': 'Additional',
        'random': 'Random',
    }
    method_naming = {
        'shine': 'SHINE with Adjoint Broyden',
        'fpn': 'Jacobian-Free method',
    }
    for direction, direction_results in inv_quality_results.items():
        print(direction)
        for i_method, (method, method_results) in enumerate(direction_results.items()):
            ax = allaxes[i_method]
            ax.scatter(
                # 0 rdiff, 1 ratio, 2 correl
                method_results['ratio'],
                method_results['correl'],
                label=naming[direction],
                s=3.,
                **styles[direction],
            )
            ax.set_title(method_naming[method])
            # ax.set_ylim([0.74, 0.94])
            # ax.set_xlim([1.1, 1.4])
            if method == 'shine':
                ax.set_ylabel(r'$\operatorname{cossim}(a, b)$')
            ax.set_xlabel(r'$\|a \|/\| b \|$')
            median_correl = np.median(method_results['correl'])
            median_ratio = np.median(method_results['ratio'])
            print(method, median_correl, median_ratio)
    handles, labels = ax.get_legend_handles_labels()

    ### legend
    ax_legend = allaxes[-1]
    legend = ax_legend.legend(
        handles,
        labels,
        loc='center',
        ncol=1,
        handlelength=1.5,
        handletextpad=.2,
        title=r'\textbf{Direction}',
    )
    ax_legend.axis('off')
    fig_name = 'adj_broyden_inversion'
    if opa:
        fig_name += '_opa'
    if not random_prescribed:
        fig_name += '_true_grad'
    fig_name += f'scatter_{dataset}_{model_size}.pdf'
    plt.savefig(fig_name, dpi=300)


def save_results(
        n_runs=100,
        random_prescribed=False,
        dataset='imagenet',
        model_size='SMALL',
):
    for opa in [False, True]:
        print('='*20)
        if opa:
            print('With OPA')
        else:
            print('Without OPA')
        inv_quality_results = adj_broyden_correl(
            opa,
            n_runs,
            random_prescribed,
            dataset,
            model_size,
        )
        res_name = f'adj_broyden_inv_results_{dataset}_{model_size}'
        if opa:
            res_name += '_opa'
        if not random_prescribed:
            res_name += '_true_grad'
        res_name += '.pkl'
        with open(res_name, 'wb') as f:
            pickle.dump(inv_quality_results, f)


if __name__ == '__main__':
    n_runs = 100
    random_prescribed = False
    save_results = False
    reload_results = False
    plot_results = True
    dataset = 'imagenet'
    model_size = 'SMALL'
    print('Ratio is true inv over approx inv')
    print('Results are presented: method, median correl, median ratio')
    for opa in [False, True]:
        print('='*20)
        if opa:
            print('With OPA')
        else:
            print('Without OPA')
        res_name = f'adj_broyden_inv_results_{dataset}_{model_size}'
        if opa:
            res_name += '_opa'
        if not random_prescribed:
            res_name += '_true_grad'
        res_name += '.pkl'
        with open(res_name, 'rb') as f:
            inv_quality_results = pickle.load(f)
        present_results(
            inv_quality_results,
            opa=opa,
            random_prescribed=random_prescribed,
            dataset=dataset,
            model_size=model_size,
        )
        print('='*20)
