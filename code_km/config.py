#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Experiment Configuration.
"""
from os.path import join, abspath, dirname
import torch


ROOT = abspath(dirname(__file__))


class Config:

    def __init__(self,
                 # info parameters
                 mode='train',
                 model_type='bart-large',  # 'bart-large' | 'bart' | 'bart-comet' | 't5' | 'gpt2'
                 task='predict-tail',
                 data_name='iekg',
                 split_method='relation',  # 'relation' | 'idiom'
                 # training parameters
                 eval_metric='rouge',  # 'rouge' or 'bleu' (determines which is used to save the best checkpoints)
                 num_epoch=3,
                 batch_size=16,
                 learning_rate=1e-5,
                 weight_decay=0.0,
                 warm_up_step=0,
                 freeze_embed=False,
                 freeze_encoder=False,
                 use_gpu=True,
                 continue_train=False,
                 use_tensorboard=True,
                 load_checkpoint_type='latest',  # 'latest' or 'best'
                 # generation parameters
                 num_beams=10,
                 num_return_sequences=3,
                 top_k=0.95,
                 top_p=0.95
                 ):

        # Settings - (regularly changed)
        # ==============================================================================================================
        self.MODE = mode
        self.MODEL_TYPE = model_type
        self.DATA_NAME = data_name
        self.EVAL_METRIC = eval_metric
        self.SPLIT_METHOD = split_method
        self.MODEL_NAME = '{}_{}_{}'.format(model_type, task, data_name)
        self.PATH_TO_META_DATA = './data/meta_data_{}_{}.json'.format(task, data_name)

        self.USE_GPU = use_gpu
        self.CONTINUE_TRAIN = continue_train
        self.USE_TENSORBOARD = use_tensorboard
        self.NUM_WORKER = 0
        self.SEED = 42
        self.MAX_SEQ_LEN = 1024

        # Checkpoint management
        self.PATH_TO_CHECKPOINT = join(ROOT, 'checkpoints/{}_{}_{}_{}_{}/'.format(
            model_type, task, data_name, split_method, eval_metric,))

        self.LOAD_CHECKPOINT_TYPE = load_checkpoint_type
        # path to comet2020 BART model downloaded from AllenNLP
        self.PATH_TO_COMET_BART = '../../Comet/comet_atomic2020_bart/comet-atomic_2020_BART'

        # ++++++++++++++++++++++++++++++++++++++++++ PARAMETERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Train Parameters
        # ==============================================================================================================
        self.NUM_EPOCHS = num_epoch
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.WEIGHT_DECAY = weight_decay
        self.VALID_FREQ = 1
        self.SAVE_FREQ = 1
        self.WARM_UP_STEP = warm_up_step
        # Inference Parameters
        # ==============================================================================================================
        self.NUM_BEAMS = num_beams
        self.NUM_RETURN_SEQUENCES = num_return_sequences
        self.TOP_K = top_k
        self.TOP_P = top_p
        # Model Parameters
        # ==============================================================================================================
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and self.USE_GPU else "cpu")
        if self.USE_GPU and not torch.cuda.is_available():
            print('=>[!] NO GPU is available in this environment! Using CPU instead!')
        self.PRETRAINED_HIDDEN_SIZE = 768 if self.MODEL_TYPE not in ['bart-comet', 'bart-large'] else 1024
        self.FREEZE_EMBEDDING = freeze_embed
        self.FREEZE_ENCODER = freeze_encoder
        self.LOGGING_STEP = 1000

        # Mode based parameters
        if self.MODE == 'test' or self.MODE == 'inference':
            self.CONTINUE_TRAIN = True
            self.USE_TENSORBOARD = False

        print('Initializing Experiments...')
        print('==> Model Type: {}'.format(self.MODEL_TYPE))
        print('==> Num Epoch: {}'.format(self.NUM_EPOCHS))
        print('==> Batch Size: {}'.format(self.BATCH_SIZE))
        print('----------------------------------------------------')



