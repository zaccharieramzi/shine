# Modified based on the DEQ repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import torch.autograd as autograd
import numpy as np
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored
import copy
from mdeq_lib.modules.broyden import broyden, analyze_broyden, rmatvec
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


def norm_diff(new, old, show_list=False):
    if show_list:
        return [(new[i] - old[i]).norm().item() for i in range(len(new))]
    return np.sqrt(sum((new[i] - old[i]).norm().item()**2 for i in range(len(new))))


class DEQFunc2d(Function):
    """ Generic DEQ module that uses Broyden's method to find the equilibrium state """

    @staticmethod
    def f(func, z1, u, *args):
        return func(z1, u, *args)

    @staticmethod
    def g(func, z1, u, cutoffs, *args):
        z1_list = DEQFunc2d.vec2list(z1, cutoffs)
        return DEQFunc2d.list2vec(DEQFunc2d.f(func, z1_list, u, *args)) - z1

    @staticmethod
    def list2vec(z1_list):
        bsz = z1_list[0].size(0)
        return torch.cat([elem.reshape(bsz, -1, 1) for elem in z1_list], dim=1)

    @staticmethod
    def vec2list(z1, cutoffs):
        bsz = z1.shape[0]
        z1_list = []
        start_idx, end_idx = 0, cutoffs[0][0] * cutoffs[0][1] * cutoffs[0][2]
        for i in range(len(cutoffs)):
            z1_list.append(z1[:, start_idx:end_idx].view(bsz, *cutoffs[i]))
            if i < len(cutoffs)-1:
                start_idx = end_idx
                end_idx += cutoffs[i + 1][0] * cutoffs[i + 1][1] * cutoffs[i + 1][2]
        return z1_list

    @staticmethod
    def broyden_find_root(func, z1, u, eps, *args):
        bsz = z1[0].size(0)
        z1_est = DEQFunc2d.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        threshold, train_step, writer = args[-3:]

        g = lambda x: DEQFunc2d.g(func, x, u, cutoffs, *args)
        result_info = broyden(g, z1_est, threshold=threshold, eps=eps, name="forward")
        z1_est = result_info['result']
        nstep = result_info['nstep']
        lowest_step = result_info['lowest_step']
        diff = result_info['diff']
        r_diff = min(result_info['new_trace'][1:])

        if z1_est.get_device() == 0:
            if writer is not None:
                writer.add_scalar('forward/diff', result_info['diff'], train_step)
                writer.add_scalar('forward/nstep', result_info['nstep'], train_step)
                writer.add_scalar('forward/lowest_step', result_info['lowest_step'], train_step)
                writer.add_scalar('forward/final_trace', result_info['new_trace'][lowest_step], train_step)

        status = analyze_broyden(result_info, judge=True)
        if status:
            err = {"z1": z1}
            analyze_broyden(result_info, err=err, judge=False, name="forward", save_err=False)

        if threshold > 30:
            torch.cuda.empty_cache()
        return DEQFunc2d.vec2list(z1_est.clone().detach(), cutoffs), result_info

    @staticmethod
    def forward(ctx, func, z1, u, *args):
        nelem = sum([elem.nelement() for elem in z1])
        eps = 1e-5 * np.sqrt(nelem)
        ctx.args_len = len(args)
        with torch.no_grad():
            z1_est, result_info = DEQFunc2d.broyden_find_root(func, z1, u, eps, *args)  # args include pos_emb, threshold, train_step
            Us = result_info['Us']
            VTs = result_info['VTs']
            nstep = result_info['lowest_step']
            qN_tensors = (Us, VTs, torch.tensor(nstep))
            # If one would like to analyze the convergence process (e.g., failures, stability), should
            # insert here or in broyden_find_root.
            return tuple(z1_est) + qN_tensors

    @staticmethod
    def backward(ctx, grad_z1, _grad_qN_tensors):
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, grad_z1, None, *grad_args)


class DEQModule2d(nn.Module):
    def __init__(self, func, func_copy, shine=False, fpn=False, gradient_correl=False):
        super(DEQModule2d, self).__init__()
        self.func = func
        self.func_copy = func_copy
        self.shine = shine
        self.fpn = fpn
        self.gradient_correl = gradient_correl

    def forward(self, z1s, us, z0, **kwargs):
        raise NotImplemented

    class Backward(Function):

        @staticmethod
        def forward(ctx, func_copy, z1, u, *args):
            ctx.save_for_backward(z1)
            ctx.u = u
            ctx.func = func_copy
            ctx.args = args
            return z1.clone()

        @staticmethod
        def backward(ctx, grad):
            # grad should have dimension (bsz x d_model x seq_len)
            bsz, d_model, seq_len = grad.size()
            # grad is basically dl/dz^star
            grad = grad.clone()
            # z1 is the predicted fixed point
            z1, = ctx.saved_tensors
            # u is the injection
            u = ctx.u
            factor = sum(ue.nelement() for ue in u) // z1.nelement()
            cutoffs = [(elem.size(1) // factor, elem.size(2), elem.size(3)) for elem in u]
            args = ctx.args
            threshold, train_step, writer, qN_tensors, shine, fpn, gradient_correl, gradient_ratio = args[-8:]
            Us, VTs, nstep = qN_tensors
            if shine:
                # TODO: allow to use Us and VTs as initialization for the backward

                dl_df_est = - rmatvec(Us[:,:,:,:nstep], VTs[:,:nstep], grad)
            elif fpn:
                dl_df_est = grad
            if not(shine or fpn) or gradient_correl or gradient_ratio:
                # here func is the mdeq module, that is the function defining the fixed point
                if gradient_correl or gradient_ratio:
                    dl_df_est_old = dl_df_est
                func = ctx.func
                z1_temp = z1.clone().detach().requires_grad_()
                u_temp = [elem.clone().detach() for elem in u]
                args_temp = args[:-1]

                with torch.enable_grad():
                    # this allows to compute the jacobian of the function
                    # g defined in the paper
                    # indeed here DEQFunc2d.g is not the application of func
                    # but rather its root finding equivalent
                    y = DEQFunc2d.g(func, z1_temp, u_temp, cutoffs, *args_temp)

                def g(x):
                    # this g here is supposed to be the function defining the
                    # root finding problem in eq 4 of the MDEQ paper
                    # recall that y here is z^star in the paper
                    # x is the x of the paper
                    y.backward(x, retain_graph=True)  # Retain for future calls to g
                    # z1_temp.grad is the vector-Jacobian product obtained via
                    # automatic-differentiation
                    # res is therefore the new estimate of x
                    res = z1_temp.grad + grad
                    z1_temp.grad.zero_()
                    return res

                eps = 2e-10 * np.sqrt(bsz * seq_len * d_model)
                dl_df_est = torch.zeros_like(grad)

                result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
                # dl_df_est is the approximation of the first part of the derivation
                # eq 3: it's dl/dz^star * (-Jg^-1)
                # which is why it's called dl / df where f is the function f of the
                # paper
                dl_df_est = result_info['result']
                nstep = result_info['nstep']
                lowest_step = result_info['lowest_step']

                if gradient_correl:
                    # compute correlation between the gradients
                    correl = torch.dot(
                        torch.flatten(dl_df_est),
                        torch.flatten(dl_df_est_old),
                    )
                    scaling = torch.norm(dl_df_est) * torch.norm(dl_df_est_old)
                    correl = correl / scaling

                if gradient_ratio:
                    ratio = torch.norm(dl_df_est) / torch.norm(dl_df_est_old)

                if gradient_correl or gradient_ratio:
                    # re-using the originally computed gradient to follow the
                    # accelerated method direction
                    dl_df_est = dl_df_est_old

                if dl_df_est.get_device() == 0:
                    if writer is not None:
                        if gradient_correl or gradient_ratio:
                            if gradient_correl:
                                writer.add_scalar('backward/correl', correl, train_step)
                            if gradient_ratio:
                                writer.add_scalar('backward/ratio', ratio, train_step)
                        else:
                            writer.add_scalar('backward/diff', result_info['diff'], train_step)
                            writer.add_scalar('backward/nstep', result_info['nstep'], train_step)
                            writer.add_scalar('backward/lowest_step', result_info['lowest_step'], train_step)
                            writer.add_scalar('backward/final_trace', result_info['new_trace'][lowest_step], train_step)



                status = analyze_broyden(result_info, judge=True)
                if status:
                    err = {"z1": z1}
                    analyze_broyden(result_info, err=err, judge=False, name="backward", save_err=False)

                if threshold > 30:
                    torch.cuda.empty_cache()

                y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

            grad_args = [None for _ in range(len(args))]
            return (None, dl_df_est, None, *grad_args)
