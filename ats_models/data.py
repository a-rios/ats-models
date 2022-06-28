#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

"""

import torch
from torch.utils.data import DataLoader, Dataset
import re
from typing import Optional, List
from transformers import MBartTokenizer

class CustomDataset(Dataset):
    def __init__(self,
                 src_file:str,
                 tgt_file: Optional[str],
                 name: str,
                 tokenizer: MBartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024,
                 src_lang: Optional[str]=None,
                 tgt_lang: Optional[str]=None,
                 tags_included: bool=False):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tags_included = tags_included

        with open(src_file, 'r') as f:
            self.inputs =  f.readlines()
        with open(tgt_file, 'r') as f:
            self.labels =  f.readlines()
        assert len(self.inputs) == len(self.labels), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.labels)}"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.inputs[idx]
        target = self.labels[idx]

        ## mbart tokenizer implementation expects only a single source and target language, so we have to do an ugly workaround here
        ## source needs to be src_lang x x x </s> src_lang
        ## target needs to be x x x </s> tgt_lang -> this can be done with as_target_tokenizer, but only with language codes in MBartTokenizer.lang_code_to_id and only one target language per tokenizer
        ## we just use the source tokenization and move the first token (tgt_lang) to the end
        ## -> tgt_lang this will later be shifted to the first position in transformers.modeling.mbart.shift_tokens_right
        ## NOTE: only works if source tag is in MBartTokenizer.lang_code_to_id, otherwise would need a dummy code for source as well!

        if self.tags_included:
            src_lang, source = self._get_tag(source)
            tgt_lang, target = self._get_tag(target)
        else:
            src_lang = self.src_lang
            tgt_lang = self.tgt_lang #TODO: option to set either source or target language independently

        self.tokenizer.src_lang= src_lang
        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        output_ids = self.tokenizer(target, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze()
        output_ids = output_ids['input_ids'].squeeze()
        output_ids[-1] = self.tokenizer.convert_tokens_to_ids(tgt_lang) # exchange src_tag with trg_tag

        return input_ids, output_ids

    def _get_tag(self, sample: str):
        m = re.search('^(.._..)\s(.*)', sample)
        if m:
            tag = m.group(1)
            line = m.group(2)
        else:
            print(f"No tag found in line {sample} when --tags_included was set.")
        return tag, line

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids

    def prepare_input(input_ids, pad_token_id):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == pad_token_id] = 0
        return input_ids, attention_mask


class CustomDatasetForInference(CustomDataset):
    def __init__(self,
                 src_file:str,
                 tgt_file: Optional[str],
                 name: str,
                 tokenizer: MBartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024,
                 src_lang: Optional[str]=None,
                 tgt_lang: Optional[str]=None,
                 tags_included: bool=False,
                 target_tags: Optional[str]=None):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tags_included = tags_included
        self.reference = None
        self.decoder_start_tokens = None

        with open(src_file, 'r') as f:
            self.inputs =  f.readlines()
        if tgt_file is not None:
            with open(tgt_file, 'r') as f:
                self.reference =  f.readlines()
            assert len(self.inputs) == len(self.reference), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.reference)}"
        if target_tags is not None:
            with open(self.target_tags, 'r') as f:
                self.decoder_start_tokens =  f.readlines()
            assert len(self.inputs) == len(self.decoder_start_tokens), f"Source and target tags have different length, need one target language tag per input sample: {len(self.inputs)} vs. {len(self.decoder_start_tokens)}"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.inputs[idx]
        decoder_start_token = None
        reference = None
        decoder_start_token = self.decoder_start_tokens[idx] if self.decoder_start_tokens is not None else None

        if self.reference is not None:
            reference = self.reference[idx]

            if self.tags_included:
                src_lang, source = self._get_tag(source)
                decoder_start_token, target = self._get_tag(reference)

        self.tokenizer.src_lang= src_lang
        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze()

        output_ids = None
        if target is not None:
            output_ids = self.tokenizer(target, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
            output_ids = output_ids['input_ids'].squeeze()
            output_ids[-1] = self.tokenizer.convert_tokens_to_ids(decoder_start_token) # exchange src_tag with trg_tag

        return input_ids, output_ids, decoder_start_token

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, ref, target_tags = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        if ref is not None:
            ref = torch.nn.utils.rnn.pad_sequence(ref, batch_first=True, padding_value=pad_token_id)
        return input_ids, ref, target_tags
