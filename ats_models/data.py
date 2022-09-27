#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

prepare_long_input() adapted from:
    https://github.com/allenai/longformer/
"""

import torch
from torch.utils.data import DataLoader, Dataset
import re
import json
import logging
from typing import Optional, List
from transformers import MBartTokenizer
from .longmbart.longformer_enc_dec import MLongformerEncoderDecoderForConditionalGeneration
from .longmbart.sliding_chunks import pad_to_window_size
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
                 src_tags_included: bool=False,
                 tgt_tags_included: bool=False):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tags_included = src_tags_included
        self.tgt_tags_included = tgt_tags_included

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
        src_lang = self.src_lang
        tgt_lang = self.tgt_lang

        if self.src_tags_included:
            src_lang, source = self._get_tag(source)
        if self.tgt_tags_included:
            tgt_lang, target = self._get_tag(target)

        assert src_lang is not None, "Source language tag needed: Either use --src_tags_included with input text where the first token in each line is the language tag, or use --src_lang to set the source language globally for all samples."
        self.tokenizer.src_lang= src_lang

        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        labels = self.tokenizer(target, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze()
        labels = labels['input_ids'].squeeze()

        # can't use shift_tokens_right from modeling_mbart with batch_sizes > 1, does not take padding into account. prepare sequences here, without padding
        # input_ids: tokens, eos (2), lang_id
        # from labels create:
        # decoder_input = lang_id tokens
        # labels = tokens eos (2)
        decoder_input_ids = torch.roll(labels, shifts=1) # move lang_id to first position
        decoder_input_ids[0] = self.tokenizer.convert_tokens_to_ids(tgt_lang) # exchange src_tag with trg_tag
        decoder_input_ids = decoder_input_ids[:-1] # cut off eos
        labels= labels[:-1] # cut off lang_id
        return input_ids, decoder_input_ids, labels

    def _get_tag(self, sample: str):
        m = re.search('^(.._[^\s\t]+)\s(.*)', sample)
        if m:
            tag = m.group(1)
            line = m.group(2)
        else:
            print(f"No tag found in line {sample} when --tags_included was set.")
            exit(1)
        return tag, line

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, decoder_input_ids, labels = list(zip(*batch))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

        return input_ids, decoder_input_ids, labels

    def prepare_input(input_ids, is_long_model, attention_mode, attention_window, pad_token_id, global_attention_indices):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == pad_token_id] = 0

        # attention longformer: 1 local, 2 global, 0 none
        if is_long_model:
            index_of_last_nonpad = (attention_mask.ne(0).sum(dim=1) - 1).squeeze(-1)
            for glob_i in global_attention_indices:
                ## negative indices: discount from index_of_last_nonpad (only need to do this if batch_size > 1, otherwise there is no padding at this point and we can just use the negative indices directly
                if glob_i < 0 and input_ids.shape[0] > 1:
                    for i, last_nonpad in enumerate(index_of_last_nonpad): # i: iterator over samples in batch
                        glob = int(last_nonpad) + glob_i +1
                        attention_mask[i][int(glob)] = 2
                # indices > 0
                else:
                    attention_mask[:, glob_i] = 2
            if attention_mode == 'sliding_chunks':
                half_padding_mod = attention_window[0]
            elif attention_mode == 'sliding_chunks_no_overlap':
                half_padding_mod = attention_window[0] / 2
            else:
                raise NotImplementedError
            input_ids, attention_mask = pad_to_window_size(input_ids, attention_mask, half_padding_mod, pad_token_id)
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
                 src_tags_included: bool=False,
                 tgt_tags_included: bool=False,
                 target_tags: Optional[str]=None):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tags_included = src_tags_included
        self.tgt_tags_included = tgt_tags_included
        self.reference = None
        self.decoder_start_tokens = None

        with open(src_file, 'r') as f:
            self.inputs =  f.readlines()
        if tgt_file is not None:
            with open(tgt_file, 'r') as f:
                self.reference =  f.readlines()
            assert len(self.inputs) == len(self.reference), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.reference)}"
        if target_tags is not None:
            with open(target_tags, 'r') as f:
                self.decoder_start_tokens =  f.readlines()
            assert len(self.inputs) == len(self.decoder_start_tokens), f"Source and target tags have different length, need one target language tag per input sample: {len(self.inputs)} vs. {len(self.decoder_start_tokens)}"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.inputs[idx]
        decoder_start_token = None
        reference = None
        target = None
        decoder_start_token = self.decoder_start_tokens[idx] if self.decoder_start_tokens is not None else self.tgt_lang
        src_lang = self.src_lang
        if self.src_tags_included:
            src_lang, source = self._get_tag(source)
        if self.reference is not None:
            reference = self.reference[idx]
            if self.tgt_tags_included:
                decoder_start_token, target = self._get_tag(reference) # if tags are included in the reference, extract and set decoder_start_token
            else:
                target = reference

        assert src_lang is not None, "Source language tag needed: Either use --src_tags_included with input text where the first token in each line is the language tag, or use --src_lang to set the source language globally for all samples."
        self.tokenizer.src_lang= src_lang

        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze()

        return input_ids, target, decoder_start_token # return reference here (string) + decoder_start_token

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, ref, target_tags = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        return input_ids, ref, target_tags

class CustomDatasetUZHJson(CustomDataset): # TODO: make this more general to work with json for other language pairs
    def __init__(self,
                 json_files:List[str],
                 name: str,
                 tokenizer: MBartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024,
                 src_lang: Optional[str] =None,
                 tgt_lang: Optional[str] = None,
                 remove_xml: Optional[bool]=False):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang =tgt_lang
        self.remove_xml = remove_xml
        self.tgt_tags_included = True # language tags are added to the tensors from the json

        self.map_lang_ids = {'a1': 'de_A1', 'a2': 'de_A2', 'b1':'de_B1'}

        self.inputs = [] # list of tuples (lang_id, sentence)
        self.labels = [] # list of tuples (lang_id, sentence)

        sample_count=0
        src_id = "de_DE" if self.src_lang is None else self.src_lang

        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                for sample in json_data['segments']:
                    source = sample['original']
                    sample_count += 1
                    tgt_lang = list(sample.keys())[0]
                    target = sample[tgt_lang]
                    if tgt_lang in self.map_lang_ids.keys():
                        tgt_lang = self.map_lang_ids[tgt_lang]
                    if self.remove_xml:
                        source, target = self._remove_xml(source, target)
                    if not (self._is_empty(source) or self._is_empty(target)): # necessary, still some noisy samples in json that consist of only xml tags without content
                        tgt_id = tgt_lang if self.tgt_lang is None else self.tgt_lang
                        self.inputs.append((src_id, source))
                        self.labels.append((tgt_id, target))
        assert len(self.inputs) == len(self.labels), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.labels)}"

        if len(self.inputs) < sample_count:
            logger.warn(f"{sample_count - len(self.inputs)} samples have been omitted from the {name} json dataset because either source or target was an empty string.")

    def __getitem__(self, idx):
            """
            mbart tokenizer implementation expects only a single source and target language, so we have to do an ugly workaround here
            source needs to be src_lang x x x </s> src_lang
            target needs to be x x x </s> tgt_lang -> this can be done with as_target_tokenizer, but only with language codes in MBartTokenizer.lang_code_to_id and only one target language per tokenizer
            we just use the source tokenization and move the first token (tgt_lang) to the end
            NOTE: only works if source tag is in MBartTokenizer.lang_code_to_id, otherwise would need a dummy code for source as well!
            """
            (src_lang, source) = self.inputs[idx]
            (tgt_lang, target) = self.labels[idx]

            self.tokenizer.src_lang= src_lang

            input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
            labels = self.tokenizer(target, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
            input_ids = input_ids['input_ids'].squeeze()
            labels = labels['input_ids'].squeeze()

            # can't use shift_tokens_right from modeling_mbart with batch_sizes > 1, does not take padding into account. prepare sequences here, without padding
            # input_ids: tokens, eos (2), lang_id
            # from labels create:
            # decoder_input = lang_id tokens
            # labels = tokens eos (2)
            decoder_input_ids = torch.roll(labels, shifts=1) # move lang_id to first position
            decoder_input_ids[0] = self.tokenizer.convert_tokens_to_ids(tgt_lang) # exchange src_tag with trg_tag
            decoder_input_ids = decoder_input_ids[:-1] # cut off eos
            labels= labels[:-1] # cut off lang_id
            return input_ids, decoder_input_ids, labels


    def _remove_xml(self, source, target):
        src_soup = BeautifulSoup(source, "html.parser")
        trg_soup = BeautifulSoup(target, "html.parser")
        return src_soup.text, trg_soup.text

    def _is_empty(self, string):
        return re.match('^[\s\t\n]*$', string)
