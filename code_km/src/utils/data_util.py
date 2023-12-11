#!/usr/bin/env python
# -*- coding:utf-8 -*-\
from transformers import BartTokenizerFast, T5TokenizerFast, GPT2TokenizerFast
from torch.utils import data as torch_data
from src.utils.file_util import load_json_file


class Dataset(torch_data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, xs):
        super(Dataset, self).__init__()
        self.xs = xs
        self.num_total_seqs = len(self.xs)

    def __len__(self):
        return self.num_total_seqs

    def __getitem__(self, index):
        return self.xs[index]


class DataHandler(object):

    def __init__(self, config):
        super(DataHandler, self).__init__()
        self.config = config
        if self.config.MODEL_TYPE == 'bart':
            self.tokenizer = BartTokenizerFast.from_pretrained(
                'facebook/bart-base',
            )
        elif self.config.MODEL_TYPE == 'bart-large':
            self.tokenizer = BartTokenizerFast.from_pretrained(
                'facebook/bart-large',
            )
        elif self.config.MODEL_TYPE == 'bart-comet':
            self.tokenizer = BartTokenizerFast.from_pretrained(
                self.config.PATH_TO_COMET_BART,
            )
        elif self.config.MODEL_TYPE == 't5':
            self.tokenizer = T5TokenizerFast.from_pretrained(
                't5-base',
            )
        elif self.config.MODEL_TYPE == 'gpt2':
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                'gpt2',
            )
        else:
            raise NotImplementedError('The Model Type [{}] is not implemented!'.format(self.config.MODEL_TYPE))

        self.load_data()
        self.init_generators()
        self.update_config()

    def load_data(self):
        print('\n* Load data and tokenizer...')
        path_to_data_files = load_json_file(self.config.PATH_TO_META_DATA)
        additional_tokens_list = load_json_file(path_to_data_files['additional_tokens_list'])
        if self.config.MODEL_TYPE != 'bart-comet':
            num_added_toks = self.tokenizer.add_tokens(additional_tokens_list)
            print('==> [{}] Special Tokens added to tokenizer!'.format(num_added_toks))
        if self.config.MODEL_TYPE == 'gpt2':
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})
            print('===> Added [PAD] and [EOS] token for GPT-2 model...')
        load_path = path_to_data_files['path_to_data'].format(self.config.SPLIT_METHOD)
        self.raw_data = load_json_file(load_path)
        print('==> Loaded data from : {}'.format(load_path))

    def init_generators(self):
        if self.config.MODE == 'train':

            self.train_dataset = Dataset(self.raw_data['train'])
            self.trainset_generator = torch_data.DataLoader(self.train_dataset,
                                                            batch_size=self.config.BATCH_SIZE,
                                                            collate_fn=self.collate_fn,
                                                            shuffle=True,
                                                            num_workers=self.config.NUM_WORKER,
                                                            drop_last=True)
            # data loader for validset
            self.valid_dataset = Dataset(self.raw_data['test'])
            self.validset_generator = torch_data.DataLoader(self.valid_dataset,
                                                            batch_size=self.config.BATCH_SIZE,
                                                            collate_fn=self.collate_fn,
                                                            shuffle=False,
                                                            num_workers=self.config.NUM_WORKER,
                                                            drop_last=False)
        else:
            self.test_dataset = Dataset(self.raw_data['test'])
            self.testset_generator = torch_data.DataLoader(self.test_dataset,
                                                           batch_size=self.config.BATCH_SIZE,
                                                           collate_fn=self.collate_fn,
                                                           shuffle=False,
                                                           num_workers=self.config.NUM_WORKER,
                                                           drop_last=False)

    def update_config(self):
        def get_batch_size(dataset_size):
            if dataset_size % self.config.BATCH_SIZE == 0:
                return dataset_size // self.config.BATCH_SIZE
            else:
                return dataset_size // self.config.BATCH_SIZE + 1

        # training parameters
        if self.config.MODE == 'train':
            self.config.train_size = len(self.train_dataset)
            self.config.valid_size = len(self.valid_dataset)
            print('===> Training dataset size: {}'.format(self.config.train_size))
            print('===> Validation dataset size: {}'.format(self.config.valid_size))
            self.config.num_batch_train = get_batch_size(self.config.train_size)
            self.config.num_batch_valid = get_batch_size(self.config.valid_size)
            self.config.NUM_TRAIN_STEPS = self.config.num_batch_train * self.config.BATCH_SIZE
        else:
            self.config.test_size = len(self.test_dataset)
            print('===> Testing dataset size: {}'.format(self.config.test_size))
            self.config.num_batch_test = get_batch_size(self.config.test_size)
            self.config.NUM_TRAIN_STEPS = 10000  # only a placeholder

    # Function to generate word definition embeddings from a list of word definitions

    def generate_model_inputs(self, input_sent, target_sent):
        input_encodings = self.tokenizer.batch_encode_plus(input_sent,
                                                           return_tensors='pt',
                                                           padding=True,
                                                           max_length=self.config.MAX_SEQ_LEN,
                                                           truncation=True)
        target_encodings = self.tokenizer.batch_encode_plus(target_sent,
                                                            return_tensors='pt',
                                                            padding=True,
                                                            max_length=self.config.MAX_SEQ_LEN,
                                                            truncation=True)
        labels = target_encodings['input_ids']

        return input_encodings['input_ids'], input_encodings['attention_mask'], labels

    def collate_fn(self, data):
        # 1. unpack data
        heads, relations, target_sents, _, _ = zip(*data)
        heads, relations, target_sents = list(heads), list(relations), list(target_sents)
        input_sents = []
        for i in range(len(heads)):
            input_sents.append(heads[i] + ' ' + relations[i] + ' [GEN]')

        if self.config.MODEL_TYPE == 'gpt2' and self.config.MODE == 'train':
            for i in range(len(target_sents)):
                input_sents[i] = input_sents[i] + ' ' + target_sents[i] + ' [EOS]'
                target_sents[i] = input_sents[i]

        input_ids, attention_mask, target_ids = self.generate_model_inputs(input_sents, target_sents)

        if 'bart' in self.config.MODEL_TYPE:
            y_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
            lm_labels[target_ids[:, 1:] == self.tokenizer.pad_token_id] = -100

            return {'inputs': {'input_ids': input_ids.long().to(self.config.DEVICE),
                               'attention_mask': attention_mask.long().to(self.config.DEVICE)
                               },
                    'decoder_input_ids': y_ids.long().to(self.config.DEVICE),
                    'labels': lm_labels.long().to(self.config.DEVICE),
                    'targets': target_ids.long().to(self.config.DEVICE),
                    }
        else:
            return {'inputs': {'input_ids': input_ids.long().to(self.config.DEVICE),
                               'attention_mask': attention_mask.long().to(self.config.DEVICE)
                               },
                    'labels': target_ids.long().to(self.config.DEVICE),
                    'targets': target_ids.long().to(self.config.DEVICE)
                    }
