#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is adapted from AllenAI's Longformer:
    https://github.com/allenai/longformer/

Note:
    Authors: Annette Rios (arios@cl.uzh.ch) Tannon Kew (kew@cl.uzh.ch)

"""

import os
import argparse
import numpy as np
import json
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.plugins import DDPPlugin

from transformers import MBartTokenizer, MBartForConditionalGeneration, MBartConfig
from .longmbart.longformer_enc_dec import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig
import datasets
from typing import Optional
from functools import partial

from .data import CustomDatasetForInference
from .finetune_mbart import MBartTrainer, remove_special_tokens
from .metrics import label_smoothed_nll_loss, get_eval_scores

class Inference(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = MBartTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)

        if self.args.is_long:
            self.config = MLongformerEncoderDecoderConfig.from_pretrained(self.args.model_path)
            self.model = MLongformerEncoderDecoderForConditionalGeneration.from_pretrained(self.args.model_path, config=self.config)
        else:
            self.config = MBartConfig.from_pretrained(self.args.model_path)
            self.model = MBartForConditionalGeneration.from_pretrained(self.args.model_path, config=self.config)

        self.max_input_len = self.args.max_input_len if self.args.max_input_len is not None else self.config.max_encoder_position_embeddings
        self.max_output_len = self.args.max_output_len if self.args.max_output_len is not None else self.config.max_decoder_position_embeddings

        if args.remove_special_tokens_containing:
            print("special tokens before:", model.tokenizer.special_tokens_map)
            model.tokenizer = remove_special_tokens(model.tokenizer, args.remove_special_tokens_containing)
            print("special tokens after:", model.tokenizer.special_tokens_map)

        self.test_dataloader_object = None

    def test_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        input_ids, ref, decoder_start_tokens  = batch # ref: string; decoder_start_tokens: tgt_lang labels
        input_ids, attention_mask = CustomDatasetForInference.prepare_input(input_ids, self.args.is_long, self.config.attention_mode, self.config.attention_window, self.tokenizer.pad_token_id, self.config.global_attention_indices)
        assert (decoder_start_tokens is not None or self.test_set.tgt_lang is not None), "Need either reference with target labels or list of target labels (multilingual batches), else --tgt_lang needs to be set"

        if decoder_start_tokens is not None: # no reference but list of target language tags given
            decoder_start_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tag) for tag in decoder_start_tokens], device=input_ids.device).unsqueeze(1)
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_input_ids=decoder_start_token_ids,
                                            do_sample=self.args.do_sample,
                                            temperature=self.args.temperature,
                                            top_k=self.args.top_k,
                                            top_p=self.args.top_p,
                                            repetition_penalty=self.args.repetition_penalty,
                                            length_penalty=self.args.length_penalty,
                                            num_return_sequences=self.args.num_return_sequences,
                                            output_scores=True if self.args.output_to_json else self.args.output_scores,
                                            return_dict_in_generate=True if self.args.output_to_json else self.args.return_dict_in_generate)

        else: # no reference, need either decoder_start_tokens (--target_tags) for multilingual batches or --tgt_lang
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.max_input_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(self.testset.tgt_lang),
                                            do_sample=self.args.do_sample,
                                            temperature=self.args.temperature,
                                            top_k=self.args.top_k,
                                            top_p=self.args.top_p,
                                            repetition_penalty=self.args.repetition_penalty,
                                            length_penalty=self.args.length_penalty,
                                            num_return_sequences=self.args.num_return_sequences,
                                            output_scores=True if self.args.output_to_json else self.args.output_scores,
                                            return_dict_in_generate=True if self.args.output_to_json else self.args.return_dict_in_generate)

        if not self.args.output_to_json:
            generated_strs = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            with open(self.args.translation, 'a') as f:
                for sample in generated_strs:
                    f.write(sample + "\n")

            if self.test_set.reference is not None: # no ref = set of None
                return get_eval_scores(ref, generated_strs)
            else:
                return {'decoded' : generated_strs}

        else:
            # if running inference with self.args.batch_size
            # > 1, we need to make sure we pair the correct input sequence
            # with the correct returned hypotheses.
            batch_hyp_strs = self.tokenizer.batch_decode(generated_ids.sequences.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #TODO: fix for beam_size=1
            batch_hyp_scores = generated_ids.sequences_scores.tolist()
            batch_source_strs = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

            generated_strs = []

            for batch_i in range(len(batch_source_strs)):
                src_str = batch_source_strs[batch_i]
                if self.test_set.reference:
                    ref_str = ' '.join(ref[batch_i].split(' ')[1:]) if self.test_set.tgt_tags_included else ref[batch_i]
                else:
                    ref_str = None

                # subselect only those hyps/scores for the
                # relevant src string
                hyps = batch_hyp_strs[batch_i:batch_i+self.args.num_return_sequences]
                scores = batch_hyp_scores[batch_i:batch_i+self.args.num_return_sequences]

                output_dict = {
                    'src': src_str,
                    'ref': ref_str,
                    'hyps': [],
                    }
                # Ensure output hyps are sorted by
                # overall NLL probability scores (smaller = better).
                scored_hyps = {score: hyp for score, hyp in zip(scores, hyps)}
                for i, score in enumerate(sorted(scored_hyps.keys(), reverse=True)):
                    # add the 1-best hypothesis to generated_strs for evaluation
                    if i == 0:
                        generated_strs.append(scored_hyps[score])
                    output_dict['hyps'].append({'score': score, 'hyp': scored_hyps[score]})

                json_line = json.dumps(output_dict, ensure_ascii=False)
                with open(self.args.translation, 'a') as f:
                    f.write(json_line+"\n")

            if self.test_set.reference is not None:
                return get_eval_scores(ref, generated_strs)

            else:
                return {'decoded' : generated_strs}


    def test_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        if self.test_set.reference is not None:
            names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
            metrics = []
            for name in names:
                scores = [x[name] for x in outputs]
                metric = sum(scores)/len(scores)
                metrics.append(metric)
            logs = dict(zip(*[names, metrics]))
            print("Evaluation on provided reference [{}] ".format(self.args.test_target))
            for m,v in logs.items():
                print(f"{m}:{v}")

    def forward(self):
        pass

    def set_test_set(self, test_set: CustomDatasetForInference):
        self.test_set = test_set

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader

        sampler = torch.utils.data.distributed.DistributedSampler(self.test_set, shuffle=is_train)

        return DataLoader(self.test_set, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=partial(CustomDatasetForInference.collate_fn, pad_token_id=self.tokenizer.pad_token_id))

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--model_path", type=str, help="Path to the checkpoint directory or model name")
        parser.add_argument("--checkpoint_name", type=str, help="Checkpoint in model_path to use.")
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
        parser.add_argument("--is_long", action='store_true', help="This is a model with longformer windowed attention.")

        #data
        parser.add_argument("--test_source", type=str, default=None, help="Path to the source test file.")
        parser.add_argument("--test_target", type=str, default=None, help="Path to the target test file (optional, if given, will output rouge and bleu).")
        parser.add_argument("--target_tags", type=str, default=None, help="If test_target is not given: provide path to file with list of target tags (one per sample in test_source).")
        parser.add_argument("--src_lang", type=str, default=None, help="Source language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tgt_lang", type=str, default=None, help="Target language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tgt_tags_included", action='store_true', help="Target text files contain language tags (first token in each line).")
        parser.add_argument("--src_tags_included", action='store_true', help="Source text files contain language tags (first token in each line).")
        parser.add_argument("--max_input_len", type=int, default=512, help="maximum num of wordpieces, if unspecified, will use number of encoder positions from model config.")
        parser.add_argument("--max_output_len", type=int, default=512, help="maximum num of wordpieces, if unspecified, will use number of decoder positions from model config.")
        parser.add_argument("--remove_special_tokens_containing", type=str, nargs="+", help="Remove tokens from the special_tokens_map that contain this string")

        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator for pytorch lightning trainer (gpu or cpu).")
        parser.add_argument("--devices", type=int, nargs="+", required=True, help="Device id(s).")

        ## inference params
        parser.add_argument("--translation", type=str, default='decoded.out', help="Output file to write decoded sequence to.")
        parser.add_argument("--beam_size", type=int, default=4, help="Beam size for inference when testing/validating. Default: 4.")
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')

        parser.add_argument("--output_to_json", default=False, action="store_true", help='If true, decoding output is a verbose JSONL containing, src, tgt, and scored model output hyps')

        # decoding strategy params (passed to model.generate() (in generation_utils.py))
        parser.add_argument("--do_sample", default=False, action="store_true", help='Whether or not to use sampling ; use greedy decoding otherwise.')
        parser.add_argument("--temperature", default=1.0, type=float, help='The value used to module the next token probabilities.')
        parser.add_argument("--top_k", default=50, type=int, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
        parser.add_argument("--top_p", default=1.0, type=float, help='If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are kept for generation.')
        parser.add_argument("--repetition_penalty", default=1.0, type=float, help='The parameter for repetition penalty. 1.0 means no penalty.')
        parser.add_argument("--length_penalty", default=1.0, type=float, help='Exponential penalty to the length. 1.0 means no penalty.')
        parser.add_argument("--output_scores", default=False, action="store_true", help='Whether or not to return the prediction scores.')
        parser.add_argument("--num_return_sequences", default=1, type=int, help='The number of independently computed returned sequences for each element in the batch, i.e. N-best')
        parser.add_argument("--return_dict_in_generate", default=False, action="store_true", help='Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.')

        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")

        return parser


def main(args):

    if Path(args.translation).is_file():
        logging.info("Output file `{}` already exists and will be overwritten...".format(args.translation))
        Path(args.translation).unlink()

    checkpoint_path=os.path.join(args.model_path, args.checkpoint_name)
    inference_model = Inference.load_from_checkpoint(checkpoint_path, args=args)

    if args.print_params:
        for name, param in simplifier.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)


    test_set = CustomDatasetForInference(src_file=args.test_source,
                                         tgt_file=args.test_target,
                                         name="test",
                                         tokenizer=inference_model.tokenizer,
                                         max_input_len=args.max_input_len,
                                         max_output_len=args.max_output_len,
                                         src_lang=args.src_lang,
                                         tgt_lang=args.tgt_lang,
                                         src_tags_included=args.src_tags_included,
                                         tgt_tags_included=args.tgt_tags_included,
                                         target_tags=args.target_tags)


    inference_model.set_test_set(test_set)

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None,
                         replace_sampler_ddp=False,
                         limit_test_batches=args.test_percent_check,
                         logger=None,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         precision=32 if args.fp32 else 16, amp_backend='native',
                         )

    trainer.test(inference_model)

    print("Decoded outputs written to {}".format(args.translation))


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="Inference")
    parser = Inference.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)

