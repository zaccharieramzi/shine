import numpy as np

from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier


n_runs = 5
n_gpus = 4
n_epochs = 220
n_refines = [0, 1, 2, 5, 7, 10, None]
base_params = dict(
    model_size='LARGE',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=n_epochs,
)
parameters = []
for i_run in range(n_runs):
    base_params.update(seed=i_run)
    for n_refine in n_refines:
        base_params.update(n_refine=n_refine)
        if n_refine != 0:
            parameters += [
                dict(**base_params),
            ]
        parameters += [
            dict(shine=True, refine=True, **base_params),
            dict(fpn=True, refine=True, **base_params),
        ]

res_all = []
for params in parameters:
    train_classifier(**params)
    eval_params = dict(**params)
    eval_params.pop('n_epochs')
    metrics_names, eval_res = evaluate_classifier(**eval_params)
    res_all.append(eval_res)

# XXX: save the results in a csv
for n_refine in n_refines:
    perf_orig = [
        res for (res, params) in zip(res_all, parameters)
        if not params.get('shine', False) and not params.get('fpn', False) and params['n_refine'] == n_refine
    ]
    perf_shine = [
        res for (res, params) in zip(res_all, parameters)
        if params.get('shine', False) and params['n_refine'] == n_refine
    ]
    perf_fpn = [
        res for (res, params) in zip(res_all, parameters)
        if params.get('fpn', False) and params['n_refine'] == n_refine
    ]


    print(f'Descriptive stats for {n_refine}')
    print('Perf orig', np.mean(perf_orig), np.std(perf_orig))
    print('Perf shine', np.mean(perf_shine), np.std(perf_shine))
    print('Perf fpn', np.mean(perf_fpn), np.std(perf_fpn))
