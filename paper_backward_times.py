import argparse

import pandas as pd

from mdeq_lib.debug.cls_grad_time import train_classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train CIFAR MDEQ models with different techniques.')
    parser.add_argument('--n_gpus', '-g', default=1,
                        help='The number of GPUs to use.')
    parser.add_argument('--n_samples', '-n', default=100,
                        help='Number of samples to use for the median backward estimation. Defaults to 100.')
    parser.add_argument('--dataset', '-d', default='cifar',
                        help='Dataset to use. Defaults to CIFAR.')
    args = parser.parse_args()
    n_samples = int(args.n_samples)
    n_gpus = int(args.n_gpus)
    dataset = args.dataset
    if dataset == 'cifar':
        n_refines = [0, 1, 2, 5, 7, 10, None]
    else:
        n_refines = [0, 5, None]
    base_params = dict(
        model_size='LARGE' if dataset == 'cifar' else 'SMALL',
        dataset=dataset,
        n_gpus=n_gpus,
        n_epochs=220 if dataset == 'cifar' else 100,
        seed=0,
        gradient_correl=False,
        gradient_ratio=False,
        compute_partial=True,
        compute_total=False,
        f_thres_range=range(18, 19) if dataset == 'cifar' else range(27,28),
        n_samples=n_samples,
    )
    parameters = []
    for n_refine in n_refines:
        refine_active = n_refine is None or n_refine > 0
        if refine_active:
            base_params.update(n_refine=n_refine)
        if n_refine != 0:
            parameters += [
                dict(**base_params),
            ]
        if n_refine is not None or dataset == 'cifar':
            parameters += [
                dict(shine=True, refine=refine_active, **base_params),
                dict(fpn=True, refine=refine_active, **base_params),
            ]
    res_data = []
    for params in parameters:
        median_backward = train_classifier(**params)
        data = dict(median_backward=median_backward, **params)
        res_data.append(data)
    res_df = pd.DataFrame(res_data)
    res_df.to_csv(f'{dataset}_backward_times.csv')
