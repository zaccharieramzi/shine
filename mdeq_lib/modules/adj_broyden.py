# Modified based on the DEQ repo.

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored


from mdeq_lib.modules.broyden import _safe_norm, scalar_search_armijo, matvec, rmatvec, line_search


def adj_broyden(g, x0, threshold, eps, ls=False, name="unknown", adj_type='C'):
    """Adjoint Broyden method

    Parameters:
        - g (fun): the root defining function
        - x0 (torch.Tensor): the initial estimate for the root
        - threshold (int): the maximum number of vectors to store for the Broyden matrix
        - eps (float): the tolerance
        - ls (bool): whether to perform line search. False by default and in practice.
        - name (str): tag to check whether you are in forward or backward mode.
        - adj_type (str): whether to use the B or C type update from the paper
            "On the local convergence of adjoint Broyden methods" Schlenkrich et al. 2010
            Definition (3). For now only 'C' is implemented, the adjoint Broyden residual update.
    """
    bsz, total_hsize, n_elem = x0.size()
    dev = x0.device

    x_est = x0           # (bsz, 2d, L')
    gx = g(x_est)        # (bsz, 2d, L')
    nstep = 0
    tnstep = 0
    LBFGS_thres = min(threshold, 27)

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, n_elem, LBFGS_thres).to(dev)
    VTs = torch.zeros(bsz, LBFGS_thres, total_hsize, n_elem).to(dev)
    update = -gx
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]
    new_trace = [-1]

    # To be used in protective breaks
    protect_thres = 1e6 * n_elem
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep

    while new_objective >= eps and nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        try:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item())   # Relative residual
        except:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item() + 1e-9)
        new_trace.append(new2_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            break
        if new_objective < 3*eps and nstep > 30 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > init_objective * protect_thres:
            prot_break = True
            break

        # this is the part that changes between Broyden and Adjoint Broyden
        if adj_type != 'C':
            raise NotImplementedError('Use adj_type C for now')
        else:
            sigma = gx
        part_Us, part_VTs = Us[:,:,:,:(nstep-1)], VTs[:,:(nstep-1)]
        # a = An^{-1} sigma
        a = matvec(part_Us, part_VTs, sigma)
        # b = sigma^T g'(xn) An^{-1}
        #######
        # Backprop on g
        #######
        x_temp = x_est.clone().detach().requires_grad_()
        with torch.enable_grad():
            # NOTE: this extra call might be potentially costly
            # we could think of a way to do it in the line search, let's see
            y = g(x_temp)
        y.backward(sigma)
        b = x_temp.grad
        #######
        b = rmatvec(part_Us, part_VTs, b)
        x_temp.grad.zero_()
        # c = (sigma - b) / (b^T sigma)
        c = (sigma - b) / torch.einsum('bij, bij -> b', b, sigma)[:, None, None]
        # these next 2 assignments allow us to get back to the original writing of
        # broyden by Shaojie
        u = a
        vT = c
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,(nstep-1) % LBFGS_thres] = vT
        Us[:,:,:,(nstep-1) % LBFGS_thres] = u
        update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx)

    # NOTE: why was this present originally? is it a question of memory?
    # Us, VTs = None, None
    return {"result": lowest_xest,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace,
            "new_trace": new_trace,
            "eps": eps,
            "threshold": threshold,
            "Us": Us,
            "VTs": VTs}
