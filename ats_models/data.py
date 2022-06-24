#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

"""

import torch
from torch.utils.data import DataLoader, Dataset
import re

class CustomDataset(Dataset):
    def __init__(self, src_file, tgt_file, name, tokenizer, max_input_len, max_output_len, src_lang, tgt_lang, tags_included):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tags_included = tags_included

        with open(src_file, 'r') as f:
            self.inputs =  f.readlines()
        self.labels = None
        if tgt_file is not None:
            with open(tgt_file, 'r') as f:
                self.labels =  f.readlines()

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
            src_lang, src_line = self._get_tag(source)
            tgt_lang, tgt_line = self._get_tag(target)

        self.tokenizer.src_lang= src_lang
        input_ids = self.tokenizer(src_line, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        output_ids = self.tokenizer(tgt_line, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
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
