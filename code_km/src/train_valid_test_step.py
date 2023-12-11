#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training/validation/inference step for Seq2seq model.
"""

import torch
from tqdm import tqdm
from src.utils.eval_util import *


def train_step(model, data_handler, epoch, writer):
    """Train model for one epoch"""
    model.train()
    # performance recorders
    loss_epoch = AverageMeter()
    metric_epoch = AverageMeter()

    # train data for a single epoch
    bbar = tqdm(enumerate(data_handler.trainset_generator), ncols=100, leave=False,
                total=data_handler.config.num_batch_train)

    for idx, data in bbar:
        loss, logits = model(data)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        model.scheduler.step()
        # prepare for eval
        ys = data['targets'].cpu().detach().numpy()
        ys_ = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        batch_size = ys.shape[0]
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)
        # eval results
        output_text = convert_ids_to_clean_text(model.tokenizer, ys_)
        reference_text = convert_ids_to_clean_text(model.tokenizer, ys)
        if data_handler.config.EVAL_METRIC == 'rouge':
            metric_score = calculate_rouge(output_text, reference_text)['rougeL']
        elif data_handler.config.EVAL_METRIC == 'bleu':
            metric_score = calculate_bleu_score(output_text, reference_text)['bleu']
        else:
            raise NotImplementedError('The evaluation metric [{}] is NOT implemented!'.format(
                data_handler.config.EVAL_METRIC))
        metric_epoch.update(metric_score, batch_size)
        # set progress bar
        bbar.set_description("Phase: [Train] | Train Loss: {:.5f} | {}: {:.3f} |".format(
            loss, data_handler.config.EVAL_METRIC, metric_score))
        if idx % data_handler.config.SAVE_FREQ == 0 and data_handler.config.USE_TENSORBOARD:
            writer.add_scalar('train_loss', loss_epoch.avg, epoch*data_handler.config.num_batch_train+idx)
            writer.add_scalar('train_metric_{}'.format(data_handler.config.EVAL_METRIC),
                              metric_epoch.avg, epoch*data_handler.config.num_batch_train+idx)

    return


def valid_step(model, data_handler):
    """Valid model for one epoch"""
    model.eval()
    torch.cuda.empty_cache()
    # performance recorders
    loss_epoch = AverageMeter()
    metric_epoch = AverageMeter()

    # valid for a single epoch
    bbar = tqdm(enumerate(data_handler.validset_generator),
                ncols=100, leave=False, total=data_handler.config.num_batch_valid)
    for idx, data in bbar:
        # model forward pass
        with torch.no_grad():
            # model forward pass to compute loss
            loss, logits = model(data)
        # prepare for eval
        ys = data['targets'].cpu().detach().numpy()
        ys_ = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        batch_size = ys.shape[0]
        loss = loss.detach().cpu().item()
        loss_epoch.update(loss, batch_size)
        # eval results
        output_text = convert_ids_to_clean_text(model.tokenizer, ys_)
        reference_text = convert_ids_to_clean_text(model.tokenizer, ys)
        if data_handler.config.EVAL_METRIC == 'rouge':
            metric_score = calculate_rouge(output_text, reference_text)['rougeL']
        elif data_handler.config.EVAL_METRIC == 'bleu':
            metric_score = calculate_bleu_score(output_text, reference_text)['bleu']
        else:
            raise NotImplementedError('The evaluation metric [{}] is NOT implemented!'.format(
                data_handler.config.EVAL_METRIC))
        metric_epoch.update(metric_score, batch_size)
        # set progress bar
        bbar.set_description("Phase: [Valid] | Valid Loss: {:.5f} | {}: {:.3f} |".format(
            loss, data_handler.config.EVAL_METRIC, metric_score))

    return loss_epoch.avg, metric_epoch.avg
