# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
import torch

from mdeq_lib.core.cls_evaluate import accuracy
from mdeq_lib.modules.deq2d import DEQFunc2d


logger = logging.getLogger(__name__)


def train(
    config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
    output_dir, tb_log_dir, writer_dict, topk=(1,5), opa=False,
    indexed_dataset=False,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    jac_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    update_freq = config.LOSS.JAC_INCREMENTAL
    global_steps = writer_dict['train_global_steps'] if writer_dict is not None else 0


    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)
    if indexed_dataset:
        fixed_points = None
    for i, data in enumerate(train_loader):
        if indexed_dataset:
            input, target, index = data
        else:
            input, target = data
        # train on partial training data
        if i >= effec_batch_num:
            break

        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet
        deq_steps = global_steps - config.TRAIN.PRETRAIN_STEPS
        if deq_steps < 0:
            # We can also regularize output Jacobian when pretraining
            factor = config.LOSS.PRETRAIN_JAC_LOSS_WEIGHT
        elif epoch >= config.LOSS.JAC_STOP_EPOCH:
            # If are above certain epoch, we may want to stop jacobian regularization training
            # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
            # will be dominating and hurt performance!)
            factor = 0
        else:
            # Dynamically schedule the Jacobian reguarlization loss weight, if needed
            factor = config.LOSS.JAC_LOSS_WEIGHT + 0.1 * (deq_steps // update_freq)
        compute_jac_loss = (torch.rand([]).item() < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)

        # compute output
        if opa:
            add_kwargs = {'y': target}
        else:
            add_kwargs = {}
        if indexed_dataset:
            add_kwargs['index'] = index
        output, jac_loss = model(
            input,
            train_step=(lr_scheduler._step_count-1),
            writer=writer_dict['writer'] if writer_dict else None,
            compute_jac_loss=compute_jac_loss,
            **add_kwargs,
        )
        if indexed_dataset:
            output, y_list = output
            y_vec = DEQFunc2d.list2vec(y_list)
            if fixed_points is None:
                # we need to flatten the fixed points
                bsz, fixed_point_dim = y_vec.shape
                fixed_points = np.empty((effec_batch_num*bsz, fixed_point_dim))
            for i, y in zip(index, y_vec):
                fixed_points[i] = y
        target = target.cuda(non_blocking=True)

        loss = criterion(output, target)
        jac_loss = jac_loss.mean()

        # compute gradient and do update step
        optimizer.zero_grad()
        if factor > 0:
            (loss + factor*jac_loss).backward()
        else:
            loss.backward()
        if config['TRAIN']['CLIP'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN']['CLIP'])
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if compute_jac_loss:
            jac_losses.update(jac_loss.item(), input.size(0))


        prec1, prec5 = accuracy(output, target, topk=topk)

        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        global_steps += 1
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, effec_batch_num, batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('jac_loss', jac_losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps

    if indexed_dataset:
        return fixed_points


def validate(config, val_loader, model, criterion, lr_scheduler, epoch, output_dir, tb_log_dir,
             writer_dict=None, topk=(1,5)):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output = model(input,
                           train_step=-1)
            target = target.cuda(non_blocking=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=topk)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Error@5 {error5:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t' \
              'Accuracy@5 {top5.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1, top5=top5,
                  error1=100-top1.avg, error5=100-top5.avg)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1
        else:
            print('Valid accuracy', top1.avg)
    return top1.avg

def validate_contractivity(val_loader, model, n_iter=20):
    max_eigens = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output = model.power_iterations(input.cuda(), n_iter=n_iter)
            # measure accuracy and record loss
            max_eigens.update(output, 1)
    print('Contract', max_eigens.avg)
    return max_eigens.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
