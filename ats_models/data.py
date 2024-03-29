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
import random
from typing import Optional, List
from transformers import MBartTokenizer, BartTokenizer
from .long_models.longformer_mbart import MLongformerEncoderDecoderForConditionalGeneration
from .long_models.sliding_chunks import pad_to_window_size
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

##################
# mBART datasets #
##################

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
                self.decoder_start_tokens =  [line.rstrip() for line in f.readlines()]
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
        decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(decoder_start_token)

        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze()

        return input_ids, decoder_start_token_id, target # return reference here (string) + decoder_start_token

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, decoder_start_tokens, ref = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        decoder_start_token_ids = torch.tensor([tag_id for tag_id in decoder_start_tokens], device=input_ids.device).unsqueeze(1)

        return input_ids, decoder_start_token_ids, ref

class CustomDatasetUZHJson(CustomDataset): # TODO: make this more general to work with json for other language pairs
    def __init__(self,
                 json_files:List[str],
                 name: str,
                 tokenizer: MBartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024,
                 src_lang: Optional[str] =None,
                 tgt_lang: Optional[str] = None,
                 remove_xml: Optional[bool]=False,
                 remove_linebreaks: Optional[bool]=False):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang =tgt_lang
        self.remove_xml = remove_xml
        self.remove_linebreaks = remove_linebreaks
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
                        source, target = self._remove_xml(source, target) # TODO: should we remove linebreaks from string?
                    if self.remove_linebreaks:
                        source = source.replace('\n', ' ')
                        source = source.replace('  ', ' ')
                        source = source.strip()
                        target = target.replace('\n', ' ')
                        target = target.replace('  ', ' ')
                        target = target.strip()
                    else:
                        source = source.replace('\n', ' <capito:br/>')
                        target = target.replace('\n', ' <capito:br/>')
                        source = source.strip()
                        target = target.strip()
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
        if source is None:
            return None, BeautifulSoup(target, "html.parser").text
        else:
            src_soup = BeautifulSoup(source, "html.parser")
            trg_soup = BeautifulSoup(target, "html.parser")
            return src_soup.text, trg_soup.text

    def _is_empty(self, string):
        return re.match('^[\s\t\n]*$', string)


class CustomInferenceDatasetUZHJson(CustomDatasetUZHJson): # TODO: make this more general to work with json for other language pairs
    def __init__(self,
                 json_files:List[str],
                 name: str,
                 tokenizer: MBartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024,
                 src_lang: Optional[str] =None,
                 tgt_lang: Optional[str] = None,
                 remove_xml: Optional[bool]=False,
                 remove_linebreaks: Optional[bool]=False):
        super(CustomInferenceDatasetUZHJson, self).__init__( json_files,
                                                    name,
                                                    tokenizer,
                                                    max_input_len,
                                                    max_output_len,
                                                    src_lang,
                                                    tgt_lang,
                                                    remove_xml,
                                                    remove_linebreaks)
        self.reference = []
        for tag, text in self.labels:
            self.reference.append(text)

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
            #labels = self.tokenizer(target, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
            input_ids = input_ids['input_ids'].squeeze()
            #labels = labels['input_ids'].squeeze()
            decoder_start_token = self.tokenizer.convert_tokens_to_ids(tgt_lang)

            # can't use shift_tokens_right from modeling_mbart with batch_sizes > 1, does not take padding into account. prepare sequences here, without padding
            # input_ids: tokens, eos (2), lang_id
            # from labels create:
            # decoder_input = lang_id tokens
            # labels = tokens eos (2)
            #decoder_start_token = labels[-1:]
            #labels= labels[:-1] # cut off lang_id
            return input_ids, decoder_start_token, target

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, decoder_start_tokens, ref = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        decoder_start_token_ids = torch.tensor([tag_id for tag_id in decoder_start_tokens], device=input_ids.device).unsqueeze(1)

        return input_ids, decoder_start_token_ids, ref


###################
#  BART datasets  #
###################


class CustomBartDataset(CustomDataset):
    def __init__(self,
                 src_file:str,
                 tgt_file: Optional[str],
                 name: str,
                 tokenizer: BartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        with open(src_file, 'r') as f:
            self.inputs =  f.readlines()
        with open(tgt_file, 'r') as f:
            self.labels =  f.readlines()
        assert len(self.inputs) == len(self.labels), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.labels)}"

    def __getitem__(self, idx):
        """
        source: bos (0) x x x .. eos (2)
        target:
        """
        source = self.inputs[idx]
        target = self.labels[idx]

        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        labels = self.tokenizer(target, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze() # bos tokens eos
        labels = labels['input_ids'].squeeze() # bos tokens eos -> tokens eos

        # need to create decoder_input_ids after, shift_tokens_right as called in BartModel uses the input ids to create decoder inputs.. we need to create them from labels
        # input_ids: bos (0) tokens, eos (2), ready from tokenizer
        # labels = bos tokens eos -> tokens eos (2) +1 pad
        # decoder_input_ids: this is a weird one...
        # decoder_start_token_id in bart config is 2 (==eos), used in shift_tokens_right in forward pass.. results in 'eos bos' at start, weird?? seems to be just bos in fairseq: https://github.com/facebookresearch/fairseq/blob/5d7ed6ab4f92d20ad10f8f792b8703e260a938ac/fairseq/models/bart/hub_interface.py#L123
        # testing with only bos for now
        # decoder_input = labels: bos tokens eos
        decoder_input_ids = labels.clone()
        labels = torch.roll(labels, shifts=-1) # move bos to last position
        labels[-1] = self.tokenizer.pad_token_id
        return input_ids, decoder_input_ids, labels


    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, decoder_input_ids, labels = list(zip(*batch))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

        return input_ids, decoder_input_ids, labels


class CustomBartDatasetForInference(CustomDataset):
    def __init__(self,
                 src_file:str,
                 tgt_file: Optional[str],
                 name: str,
                 tokenizer: BartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.reference = None

        with open(src_file, 'r') as f:
            self.inputs =  f.readlines()
        if tgt_file is not None:
            with open(tgt_file, 'r') as f:
                self.reference =  f.readlines()
            assert len(self.inputs) == len(self.reference), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.reference)}"

    def __getitem__(self, idx):
        source = self.inputs[idx]
        reference = None
        if self.reference is not None:
            reference = self.reference[idx]

        input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
        input_ids = input_ids['input_ids'].squeeze()

        return input_ids, reference # return reference here (string, optional, will calculate metrics if provided)

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, ref = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        return input_ids, ref


class CustomBartDatasetUZHJson(CustomDatasetUZHJson):
    def __init__(self,
                 json_files:List[str],
                 name: str,
                 tokenizer: BartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024,
                 remove_xml: Optional[bool]=False):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.remove_xml = remove_xml

        self.inputs = [] # list of tuples (lang_id, sentence)
        self.labels = [] # list of tuples (lang_id, sentence)

        sample_count=0

        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                for sample in json_data['segments']:
                    source = sample['original']
                    sample_count += 1
                    tgt_lang = list(sample.keys())[0]
                    target = sample[tgt_lang] # TODO: not yet done, check what the key for the simplified text is in the English json
                    if self.remove_xml:
                        source, target = self._remove_xml(source, target) # TODO: should we remove linebreaks from string?
                    if not (self._is_empty(source) or self._is_empty(target)): # necessary, still some noisy samples in json that consist of only xml tags without content

                        self.inputs.append(source)
                        self.labels.append(target)
        assert len(self.inputs) == len(self.labels), f"Source and target have different number of samples: {len(self.inputs)} vs. {len(self.labels)}"

        if len(self.inputs) < sample_count:
            logger.warn(f"{sample_count - len(self.inputs)} samples have been omitted from the {name} json dataset because either source or target was an empty string.")

    def __getitem__(self, idx):
            source = self.inputs[idx]
            target = self.labels[idx]

            input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
            labels = self.tokenizer(target, return_tensors="pt", max_length=self.max_output_len, truncation=True, padding=False)
            input_ids = input_ids['input_ids'].squeeze()
            labels = labels['input_ids'].squeeze() # should be 'tokens eos(2), TODO check
            decoder_input_ids = labels.clone()
            labels = torch.roll(labels, shifts=-1) # move bos to last position
            labels[-1] = self.tokenizer.pad_token_id # replace bos with pad
            return input_ids, decoder_input_ids, labels

class CustomBartInferenceDatasetUZHJson(CustomBartDatasetUZHJson): # TODO: make this more general to work with json for other language pairs
    def __init__(self,
                 json_files:List[str],
                 name: str,
                 tokenizer: BartTokenizer,
                 max_input_len: int=1024,
                 max_output_len: int=1024,
                 remove_xml: Optional[bool]=False):
        super(CustomBartInferenceDatasetUZHJson, self).__init__( json_files,
                                                    name,
                                                    tokenizer,
                                                    max_input_len,
                                                    max_output_len,
                                                    remove_xml)
        self.reference = []
        for text in self.labels:
            self.reference.append(text)

    def __getitem__(self, idx):
            source = self.inputs[idx]
            target = self.labels[idx]

            input_ids = self.tokenizer(source, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False)
            input_ids = input_ids['input_ids'].squeeze()

            return input_ids, target

    @staticmethod
    def collate_fn(batch, pad_token_id):
        input_ids, ref = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        return input_ids, ref

###################
#  FUDGE datasets #
###################
class FudgeDatasetJson(CustomDatasetUZHJson):
    def __init__(self,
                 json_files:List[str],
                 name: str,
                 tokenizer: MBartTokenizer,
                 max_input_len: int=1024,
                 min_input_len: int=3,
                 tgt_tag: str = None,
                 remove_xml: Optional[bool]=False,
                 remove_linebreaks: Optional[bool] = False,
                 seed: Optional[int] = 42):
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.min_input_len = min_input_len
        self.tgt_tag = tgt_tag
        self.remove_xml = remove_xml
        self.remove_linebreaks = remove_linebreaks
        self.seed = seed
        self.inv_map_lang_ids = { 'de_A1' : 'a1',  'de_A2': 'a2', 'de_B1' : 'b1'}

        positive_samples = dict() # list of tuples (sentence, 1)
        negative_samples = dict() # list of tuples (sentence, 0)

        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                for sample in json_data['segments']:
                    source = sample['original'] if 'original' in sample else None
                    tgt_lang = list(sample.keys())[0]
                    target = sample[tgt_lang]
                    if self.remove_xml:
                        source, target = self._remove_xml(source, target)
                    if self.remove_linebreaks:
                        if source is not None:
                            source = source.replace('\n', ' ')
                            source = source.replace('  ', ' ')
                            source = source.strip()
                        target = target.replace('\n', ' ')
                        target = target.replace('  ', ' ')
                        target = target.strip()
                    if not (source is None or self._is_empty(source)) and len(source.split(' ')) > self.min_input_len:
                        negative_samples[source] = 0
                    if not self._is_empty(target) and len(target.split(' ')) > self.min_input_len: # necessary, still some noisy samples in json that consist of only xml tags without content
                        # if json has target level in its name, these are the positive samples if tgt_lang == given tgt_tag
                        if self.inv_map_lang_ids[self.tgt_tag] == tgt_lang:
                            positive_samples[target] = 1
                        else:
                            negative_samples[target] = 0
        # dedup
        self.negative_samples_list = []
        d = 0
        print(len(negative_samples))
        for neg_sample in negative_samples:
            if neg_sample not in positive_samples:
                self.negative_samples_list.append((neg_sample, 0))
            else:
                d +=1
        print(f"{d} duplicates: removed from negative_samples")
        self.positive_samples_list = [(pos_sample, 1) for pos_sample in positive_samples.keys()]

        # TODO limit number of negative samples to positive samples? or use all?
        self.inputs = self.positive_samples_list + self.negative_samples_list # sample here or leave it to dataloader (dataloader shuffles only train sets)?
        print(f"Created {self.name} data set with total {len(self.inputs)} samples, {len(self.positive_samples_list)} positive and {len(self.negative_samples_list)} negative.")
        # exit(0)
        # random.Random(self.seed).shuffle(self.inputs)


    def __getitem__(self, idx):
            (sample, label) = self.inputs[idx]
            input_ids = self.tokenizer(sample, return_tensors="pt", max_length=self.max_input_len, truncation=True, padding=False, add_special_tokens=False)
            input_ids = input_ids['input_ids'].squeeze()
            label = torch.tensor([label], dtype=input_ids.dtype)
            return input_ids, label

    @staticmethod
    def collate_fn(batch, pad_token_id):
        samples, labels = list(zip(*batch))
        samples = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=pad_token_id)
        # stack labels
        labels = torch.stack(labels) # shape (batch, 1)
        labels = labels.squeeze(1)
        return samples, labels
