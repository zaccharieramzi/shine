# MDEQ - SHINE

This is the second part of the code for the paper "SHINE: SHaring the INverse Estimate from the forward pass for bi-level optimization and implicit models", submitted at the 2021 NeurIPS conference.
This source code allows to reproduce the experiments on multiscale DEQs, i.e. Figure 4, and Figure E.2. in Appendix.

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

You can further indicate how many gpus to use in each script with the `--n_gpus` option.
You can find other options using the `--help` option.
Beware:
- each CIFAR training is 11hours to 15 hours long (100 of them by default)
- each ImageNet training is 3 days to 7 days long (6 of them by default)

For a practical reproduction you might want to run those in an HPC.
For a test use, you can use the `--n_runs` and `--n_refines` options.

## Reproducing Figure E.2., Quality of the inversion using OPA in DEQs
