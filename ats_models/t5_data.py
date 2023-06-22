#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

"""

import torch
from torch.utils.data import DataLoader, Dataset
import re
import json
import logging
import random
from typing import Optional, List
from transformers import T5Tokenizer, LongT5ForConditionalGeneration,  LongT5Config
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

##################
# T5 datasets #
##################

class T5Dataset(Dataset):
    def __init__(self,
                 src_file:str,
                 tgt_file: Optional[str],
                 name: str,
                 tokenizer: T5Tokenizer,
                 max_input_len: int=16384,
                 max_output_len: int=16384,
                 prefix: Optional[str]=None):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.prefix = prefix

        with open(src_file, 'r') as f:
            self.inputs =  f.readlines()
        with open(tgt_file, 'r') as f:
            self.labels =  f.readlines()
        assert len(self.inputs) == len(self.labels), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.labels)}"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        mbart tokenizer implementation expects only a single source and target language, so we have to do an ugly workaround here
        source needs to be src_lang x x x </s> src_lang
        target needs to be x x x </s> tgt_lang -> this can be done with as_target_tokenizer, but only with language codes in MBartTokenizer.lang_code_to_id and only one target language per tokenizer
        we just use the source tokenization and move the first token (tgt_lang) to the end
        NOTE: only works if source tag is in MBartTokenizer.lang_code_to_id, otherwise would need a dummy code for source as well!
        """
        source = self.inputs[idx]
        target = self.labels[idx]
        prefix = self.prefix

        if self.prefix is not None:
            source = f"{prefix} {source}"

        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        labels = self.tokenizer(target, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze()
        labels = labels['input_ids'].squeeze()

        # decoder_input_ids will be created in forward from LongT5ForConditionalGeneration from labels:
        #https://github.com/huggingface/transformers/blob/1a6fb930fbca82d80f2c1dbd2935ec5e8cdb2cdb/src/transformers/models/longt5/modeling_longt5.py#L2054
        # input_ids: prefix tokens eos (pad)
        # decoder_input_ids: pad tokens (pad)
        # labels tokens eos
        return input_ids, labels

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, labels = list(zip(*batch))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

        return input_ids, labels

    def get_attention_mask(input_ids, pad_token_id):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == pad_token_id] = 0

        return input_ids, attention_mask


class T5ForInference(T5Dataset):
    def __init__(self,
                 src_file:str,
                 tgt_file: Optional[str],
                 name: str,
                 tokenizer: T5Tokenizer,
                 max_input_len: int=16384,
                 max_output_len: int=16384,
                 prefix: Optional[str]=None):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.prefix = prefix
        self.reference = None

        with open(src_file, 'r') as f:
            self.inputs =  f.readlines()
        if tgt_file is not None:
            with open(tgt_file, 'r') as f:
                self.reference =  f.readlines()
            assert len(self.inputs) == len(self.reference), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.reference)}"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.inputs[idx]
        reference = None

        if self.reference is not None:
            reference = self.reference[idx]

        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze()

        return input_ids, reference # return reference here (string)

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, ref = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        return input_ids, ref
