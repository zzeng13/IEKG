#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Knowledge model transformers
import os
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, GPT2LMHeadModel, AutoModelForSeq2SeqLM
from transformers import get_linear_schedule_with_warmup, AdamW


class KnowledgeCompletionModel(nn.Module):
    def __init__(self, config, tokenizer):
        super(KnowledgeCompletionModel, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        # Initialize the knowledge completion model
        self.init_model()

    def save_model(self, save_type, epoch):
        # save the projection layer only
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
        }
        save_path = self.config.PATH_TO_CHECKPOINT
        # create save directory if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = self.config.PATH_TO_CHECKPOINT + '{}_model.mdl'.format(save_type)
        torch.save(state, save_path)

    def init_model(self):
        # Load pre-trained model
        print('\n* Load pretrained language model...')
        if self.config.MODEL_TYPE == 'bart':
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

        elif self.config.MODEL_TYPE == 'bart-large':
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        elif self.config.MODEL_TYPE == 'bart-comet':
            self.model = BartForConditionalGeneration.from_pretrained(self.config.PATH_TO_COMET_BART)

        elif self.config.MODEL_TYPE == 't5':
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')

        elif self.config.MODEL_TYPE == 'gpt2':
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        else:
            raise NotImplementedError('The Model Type [{}] is not implemented!'.format(self.config.MODEL_TYPE))
        print('==> A Pre-trained Transformer with the type [{}] is loaded!'.format(self.config.MODEL_TYPE))

        # Freeze embedding layer
        self.check_freeze_params()
        # Load or initialize a model, optimizer, and scheduler
        if self.config.MODEL_TYPE != 'bart-comet':
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.config.DEVICE)

        print('\n* Initialize a new model with optimizer and scheduler...')
        # Initialize optimizer and scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.LEARNING_RATE, eps=1e-8)
        self.last_epoch = -1
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.WARM_UP_STEP,
            num_training_steps=self.config.NUM_TRAIN_STEPS,
            last_epoch=self.last_epoch
        )
        print('==> New model initialized!')
        if self.config.CONTINUE_TRAIN:
            loaded = self.load_model()

    def load_model(self):
        # load the projection layer from a check point
        save_path = self.config.PATH_TO_CHECKPOINT + '{}_model.mdl'.format(self.config.LOAD_CHECKPOINT_TYPE)
        print('\n* Try continue training from a previous checkpoint... ')
        print('==> Try loading model from {}'.format(save_path))
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.last_eopch = checkpoint['epoch']
            print('==> Generation model loaded from {}'.format(save_path))
            print('==> Training resumed from a previous checkpoint at epoch [{}]!'.format(self.last_eopch))
            loaded = True
        else:
            print('==> No Valid Checkpoint is found! The Generation model is randomly initialized!')
            loaded = False
        return loaded

    def check_freeze_params(self):
        if self.config.FREEZE_EMBEDDING:
            self.freeze_embedding_params()
        if self.config.FREEZE_ENCODER:
            self.freeze_params(self.model.get_encoder())
            self.assert_all_frozen(self.model.get_encoder())
            print('==> Encoder layers are frozen!')

    def freeze_embedding_params(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
            print('==> Embedding layer [positional + token] is frozen!')
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)
            print('==> Embedding layer [token] is frozen!')

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def assert_all_frozen(self, model):
        def grad_status(model):
            return (par.requires_grad for par in model.parameters())
        model_grads = list(grad_status(model))
        n_require_grad = sum(list(map(int, model_grads)))
        npars = len(model_grads)
        assert not any(model_grads), f"{n_require_grad / npars:.1%} of {npars} weights require grad"

    def forward(self, data):
        if self.config.MODE == 'inference':
            if self.config.MODEL_TYPE == 'gpt2':
                generated_ids = self.model.generate(
                    **data['inputs'],
                    num_beams=self.config.NUM_BEAMS,
                    num_return_sequences=self.config.NUM_RETURN_SEQUENCES,
                    top_k=self.config.TOP_K,
                    top_p=self.config.TOP_P,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                generated_ids = self.model.generate(
                    **data['inputs'],
                    num_beams=self.config.NUM_BEAMS,
                    num_return_sequences=self.config.NUM_RETURN_SEQUENCES,
                    top_k=self.config.TOP_K,
                    top_p=self.config.TOP_P,
                )

            gen_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return gen_text
        else:
            if 'bart' in self.config.MODEL_TYPE:
                outputs = self.model(**data['inputs'],
                                     labels=data['labels'],
                                     decoder_input_ids=data['decoder_input_ids'])
            else:
                outputs = self.model(**data['inputs'], labels=data['labels'])
            return outputs.loss, outputs.logits
