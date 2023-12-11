#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Training script for IEKG model.
"""
import random
import numpy as np
from datetime import datetime
from tqdm import trange
from tensorboardX import SummaryWriter
from src.utils.data_util import DataHandler
from src.train_valid_test_step import train_step, valid_step
from config import Config
import torch
from torch.multiprocessing import set_start_method
from src.models.knowledge_model import KnowledgeCompletionModel


try:
    set_start_method('spawn')
except RuntimeError:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    """
    The training script for the knowledge model.
    """
    # Initialize the configuration
    config = Config()

    # Initialize a data loader
    # ---------------------------------------------------------------------------------
    data_handler = DataHandler(config)

    # Manage and initialize model
    # ---------------------------------------------------------------------------------
    # Initialize model
    epoch_start = 0  # 0
    model = KnowledgeCompletionModel(data_handler.config, data_handler.tokenizer)
    # Add Tensorboard writer
    writer = None
    if config.USE_TENSORBOARD:
        writer = SummaryWriter(log_dir='./runs/{}_{}'.format(
            config.MODEL_NAME, datetime.today().strftime('%Y-%m-%d')))
    best_valid_score = float('-inf')
    best_loss = float('inf')

    # Train model
    # ---------------------------------------------------------------------------------
    print('\n* Model training starts...')
    ebar = trange(epoch_start, config.NUM_EPOCHS, desc='EPOCH', ncols=130, leave=True)
    set_seed(config.SEED)

    for epoch in ebar:
        # Training
        train_step(model, data_handler, epoch, writer)

        # Validation
        if epoch % config.VALID_FREQ == 0:
            valid_loss, valid_score = valid_step(model, data_handler)
            if best_valid_score < valid_score:
                best_valid_score = valid_score
                # save the best model seen so far
                model.save_model("best", epoch)
            if best_loss > valid_loss:
                best_loss = valid_loss
                model.save_model("best-loss", epoch)
            if config.USE_TENSORBOARD:
                writer.add_scalar('valid_loss', valid_loss, epoch)
                writer.add_scalar('valid_metric_{}'.format(config.EVAL_METRIC), valid_score, epoch)

        # save the latest model
        model.save_model("latest_{}".format(epoch), epoch)
    return


if __name__ == '__main__':
    train()


