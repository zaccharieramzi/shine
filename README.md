# MDEQ - SHINE

This is the second part of the code for the paper "SHINE: SHaring the INverse Estimate from the forward pass for bi-level optimization and implicit models", submitted at the 2021 NeurIPS conference.
The first part of the code to reproduce the Bi-level optimizations experiments is available [here](https://github.com/zaccharieramzi/hoag/tree/shine).
This source code allows to reproduce the experiments on multiscale DEQs, i.e. Figure 4, and Figure E.2. in Appendix.
This repo is based on the original [mdeq repo](https://github.com/locuslab/mdeq) by @jerrybai1995.

## General instructions

You need Python 3.7 or above to run this code.
This code will only run on a computer equipped with a GPU.
You can then install the requirements with: `pip install -r requirements.txt`.


## Reproducing Figure 4, DEQ

You can reproduce Figure 4 of the paper with the following sequence of scripts:
```
# cifar
python paper_trainings.py
python paper_backward_times.py
# imagenet
python paper_trainings.py --dataset imagenet --n_runs 1 --refines 0,5,None
python paper_backward_times.py --dataset imagenet
python paper_plot.py
```

You can further indicate how many gpus to use in each script with the `--n_gpus` option (default for training is 4).
You can find other options using the `--help` option.
Beware:
- each CIFAR training is 11hours to 15 hours long (100 of them by default)
- each ImageNet training is 3 days to 7 days long (6 of them by default)

For a practical reproduction you might want to run those in an HPC (i.e. change line 56-60 to work with e.g. [submitit](https://github.com/facebookincubator/submitit)).
For a test use, you can use the `--n_runs` (the number of repetitions for the error bar) and `--n_refines` (the number of points on the Pareto curve) options.

You can also just do the CIFAR trainings, by simply not running the ImageNet ones.
The Figure will still be generated.

## Reproducing Figure E.2., Quality of the inversion using OPA in DEQs

You can reproduce Figure E.2. of the paper with the following script:

```
python mdeq_lib/tests/modules/adj_broyden_correl.py
```

This should take about 15 mins to run with a single GPU.
