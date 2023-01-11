#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is mostly adapted from transformers.generation_utils:
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation_utils.py

Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

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
from .long_models.longformer_mbart import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig
from .inference_mbart import Inference
import datasets
from typing import Optional, Callable, Tuple, Iterable, Union, List, Dict, Any
from collections import defaultdict
from functools import partial

from .data import CustomDatasetForInference
from .finetune_mbart import MBartTrainer, remove_special_tokens
from .metrics import label_smoothed_nll_loss, get_eval_scores

from transformers.generation_utils import GenerationMixin, BeamSearchOutput
from transformers.generation_beam_search import BeamSearchScorer, BeamScorer
from transformers.utils import ModelOutput
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers.generation_logits_process import LogitsProcessorList
from transformers.pytorch_utils import torch_int_div
import torch.distributed as dist

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Ensemble(pl.LightningModule, GenerationMixin):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.num_models = len(self.args.checkpoints)
        self.tokenizer = MBartTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)

        if self.args.is_long:
            self.config = MLongformerEncoderDecoderConfig.from_pretrained(self.args.model_path)
        else:
            self.config = MBartConfig.from_pretrained(self.args.model_path)

        shared_args = argparse.Namespace(**vars(self.args))
        delattr(shared_args, 'checkpoints')

        self.models = []
        for i in range(self.num_models): # check can we load multiple models on 1 gpu? 1 per gpu?
            model_args = argparse.Namespace(**vars(shared_args))
            setattr(model_args, 'checkpoint', self.args.checkpoints[i])

            checkpoint_path=os.path.join(args.model_path, self.args.checkpoints[i])
            inference_model = Inference.load_from_checkpoint(checkpoint_path, map_location=torch.device("cuda:" + str(self.args.devices[0])), args=model_args) # TODO iterate over devices, one gpu per model?
            inference_model.to(torch.device('cuda'))

            self.models.append(inference_model)

        self.max_length = self.args.max_output_len if self.args.max_output_len is not None else self.models[0].config.max_length # TODO add warning here
        #min_length = min_length if self.args.min_length is not None else self.models[0].config.min_length
        if args.remove_special_tokens_containing:
            print("special tokens before:", model.tokenizer.special_tokens_map)
            model.tokenizer = remove_special_tokens(model.tokenizer, args.remove_special_tokens_containing)
            print("special tokens after:", model.tokenizer.special_tokens_map)

        self.test_dataloader_object = None
        if self.args.ensemble_mode == 'linear':
            self._interpolation = self.linear_interpolation
        elif self.args.ensemble_mode == 'log_linear':
            self._interpolation = self.log_linear_interpolation
        else:
            raise ValueError()


    def test_step(self, batch, batch_nb):

        input_ids, ref, decoder_start_tokens  = batch # ref: string; decoder_start_tokens: tgt_lang labels
        input_ids, attention_mask = CustomDatasetForInference.prepare_input(input_ids, self.args.is_long, self.config.attention_mode, self.config.attention_window, self.tokenizer.pad_token_id, self.config.global_attention_indices)
        assert (decoder_start_tokens is not None or self.test_set.tgt_lang is not None), "Need either reference with target labels or list of target labels (multilingual batches), else --tgt_lang needs to be set"

        batch_size = input_ids.shape[0]

        if decoder_start_tokens is not None: # no reference but list of target language tags given (batch_size, 1)
            decoder_start_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tag) for tag in decoder_start_tokens], device=input_ids.device).unsqueeze(1)
        else: # tgt_lang set, (batch_size, 1)
            decoder_start_token_ids = torch.full( (batch_size, 1), self.tokenizer.convert_tokens_to_ids(self.test_set.tgt_lang), device=input_ids.device) # TODO test this

        model_kwargs = {'attention_mask': attention_mask,
                        'decoder_input_ids': decoder_start_token_ids,
                        'output_attentions': False,
                        'output_hidden_states': False,
                        'use_cache': True,
                        'encoder_outputs': []}

         ## adapted from transformers.generation_utils.py, lines 1312 ff. (we only do beam search for now)
        logits_processor = self._get_logits_processor(
                        repetition_penalty=self.args.repetition_penalty,
                        no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                        encoder_no_repeat_ngram_size=None,
                        input_ids_seq_length=1, # length of decoder input ids
                        encoder_input_ids=input_ids,
                        bad_words_ids=None,
                        min_length=None,
                        max_length=self.max_length,
                        eos_token_id=self.tokenizer.eos_token_id,
                        forced_bos_token_id=None,
                        forced_eos_token_id=None,
                        prefix_allowed_tokens_fn=None,
                        num_beams=self.args.beam_size,
                        num_beam_groups=None,
                        diversity_penalty=None,
                        remove_invalid_values=None,
                        exponential_decay_length_penalty=None,
                        logits_processor=[],
                        renormalize_logits=None,
                    )

        stopping_criteria = self._get_stopping_criteria(
                    max_length=self.max_length, max_time=None, stopping_criteria=[]
                )

        if self.args.num_return_sequences > self.args.beam_size:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=self.args.beam_size,
                device=input_ids.device,
                length_penalty=self.args.length_penalty,
                do_early_stopping=False,
                num_beam_hyps_to_keep=self.args.num_return_sequences,
            )
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_ensemble_generation(
                input_ids, model_kwargs, model_input_name='input_ids'
            )

        decoder_input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(self.args.tgt_lang) if self.args.tgt_lang is not None else None,
                bos_token_id=None,
                model_kwargs=model_kwargs,
                device=input_ids.device,
            )
        decoder_input_ids, model_kwargs = self._expand_inputs_for_ensemble_generation(
                decoder_input_ids, expand_size=self.args.beam_size, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )

        generated_ids = self.ensemble_beam_search(
                decoder_input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=self.args.return_dict_in_generate,
                synced_gpus=True,
                **model_kwargs,
            )

        generated_strs = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        with open(self.args.translation, 'a') as f:
            for sample in generated_strs:
                f.write(sample + "\n")

        if self.test_set.reference is not None: # no ref = set of None
            return get_eval_scores(ref, generated_strs)
        else:
            return {'decoded' : generated_strs}

    def _prepare_encoder_decoder_kwargs_for_ensemble_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
        ) -> Dict[str, Any]:

            # 2. prepare encoder args and encoder kwargs from model kwargs
            irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "encoder_outputs"]
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not any(argument.startswith(p) for p in irrelevant_prefix)
            }

            # 3. make sure that encoder returns `ModelOutput`
            model_input_name = model_input_name if model_input_name is not None else self.main_input_name
            encoder_kwargs["return_dict"] = True
            encoder_kwargs[model_input_name] = inputs_tensor
            for inference_obj in self.models:
                encoder = inference_obj.model.get_encoder()
                enc_outputs: ModelOutput = encoder(**encoder_kwargs)
                model_kwargs["encoder_outputs"].append(enc_outputs)
            return model_kwargs

    def _expand_inputs_for_ensemble_generation(self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[List[ModelOutput]] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        encoder_outputs_with_last_hidden = []
        for i in range(self.num_models):
            ith_encoder_outputs = encoder_outputs[i]
            ith_encoder_outputs["last_hidden_state"] = ith_encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(ith_encoder_outputs.last_hidden_state.device)
            )
            encoder_outputs_with_last_hidden.append(ith_encoder_outputs)
        model_kwargs["encoder_outputs"] = encoder_outputs_with_last_hidden
        return input_ids, model_kwargs

    ## code from sockeye.beam_search
    def linear_interpolation(self, predictions):
        return (self.average_tensors(predictions).log())

    def log_linear_interpolation(self, predictions):
        log_probs = self.average_tensors([p.log() for p in predictions])
        return (log_probs.log_softmax())

    def average_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the element-wise average of a list of tensors of the same shape.

        :param tensors: A list of input tensors with the same shape.
        :return: The average of the tensors on the same device as tensors[0].
        """
        if not tensors:
            raise ValueError("tensors is empty.")
        if len(tensors) == 1:
            return tensors[0]
        assert ( all(tensors[0].shape == t.shape for t in tensors) ), "tensor shapes for averaging probabilities do not match"
        return sum(tensors) / len(tensors)  # type: ignore

    @staticmethod
    def _update_model_kwargs_for_ensemble_generation(
        outputs: List[ModelOutput], model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:

        pasts = []
        for output in outputs:
            # update past
            if "past_key_values" in output:
                pasts.append(output.past_key_values)
            elif "mems" in outputs:
                pasts.append(output.mems)
            elif "past_buckets_states" in outputs:
                pasts.append(output.past_buckets_states)
            else:
                pasts.append(None)
        model_kwargs['past'] = pasts

        return model_kwargs

    # code adapted from transformers.generation_utils.beam_search
    def ensemble_beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
        ) -> Union[BeamSearchOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation. (These are the decoder input_ids).
                beam_scorer (`BeamScorer`):
                    An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                    sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
                logits_processor (`LogitsProcessorList`, *optional*):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`, *optional*):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                max_length (`int`, *optional*, defaults to 20):
                    **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                    tokens. The maximum length of the sequence to be generated.
                pad_token_id (`int`, *optional*):
                    The id of the *padding* token.
                eos_token_id (`int`, *optional*):
                    The id of the *end-of-sequence* token.
                output_attentions (`bool`, *optional*, defaults to `False`):
                    Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                    returned tensors for more details.
                output_hidden_states (`bool`, *optional*, defaults to `False`):
                    Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                    for more details.
                output_scores (`bool`, *optional*, defaults to `False`):
                    Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
                return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                    Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
                synced_gpus (`bool`, *optional*, defaults to `False`):
                    Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
                `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            ```"""
            # init values
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
            if max_length is not None:
                warnings.warn(
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                    UserWarning,
                )
                stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
            if len(stopping_criteria) == 0:
                warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
            pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
            eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
            output_scores = output_scores if output_scores is not None else self.config.output_scores
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict_in_generate = (
                return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
            )

            batch_size = len(beam_scorer._beam_hyps)
            num_beams = beam_scorer.num_beams

            batch_beam_size, cur_len = input_ids.shape

            if num_beams * batch_size != batch_beam_size:
                raise ValueError(
                    f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
                )

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            beam_indices = (
                tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
            )
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = []
                encoder_hidden_states = []
                for i in range(self.num_models):
                    encoder_attentions.append(model_kwargs["encoder_outputs"][i].get("attentions") if output_attentions else None)
                    encoder_hidden_states.append( (
                        model_kwargs["encoder_outputs"][i].get("hidden_states") if output_hidden_states else None
                    ))

            # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
            # of the first beam are considered to avoid sampling the exact same tokens across all beams.
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))

            this_peer_finished = False  # used by synced_gpus only
            while True:

                if synced_gpus:
                    # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                    # The following logic allows an early break if all peers finished generating their sequence
                    this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                    # send 0.0 if we finished, 1.0 otherwise
                    dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                    # did all peers finish? the reduced sum will be 0.0 then
                    if this_peer_finished_flag.item() == 0.0:
                        break

                outputs = []
                for i, inference_obj in enumerate(self.models):
                    ith_model_kwargs = model_kwargs.copy()
                    ith_model_kwargs['encoder_outputs'] = model_kwargs['encoder_outputs'][i]
                    if 'past' in ith_model_kwargs:
                        ith_model_kwargs['past'] = model_kwargs['past'][i]
                    model_inputs = inference_obj.model.prepare_inputs_for_generation(input_ids, **ith_model_kwargs)
                    outputs.append(inference_obj.model(
                        **model_inputs,
                        #encoder_outputs=model_kwargs['encoder_outputs'][i],
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        )
                    )

                if synced_gpus and this_peer_finished:
                    cur_len = cur_len + 1
                    continue  # don't waste resources running the code we don't need

                probs = [ o['logits'][:, -1, :].softmax(dim=-1) for o in outputs ]
                next_token_scores = self._interpolation(probs)

                next_token_scores_processed = logits_processor(input_ids, next_token_scores)
                next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores_processed,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # reshape for beam searchprep
                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )

                next_indices = torch_int_div(next_tokens, vocab_size)
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=beam_indices,
                )

                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

                # add 'past'
                model_kwargs = self._update_model_kwargs_for_ensemble_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                if model_kwargs["past"] is not None:
                    pasts = []
                    for i, inference_obj in enumerate(self.models):
                        pasts.append(inference_obj.model._reorder_cache(model_kwargs["past"][i], beam_idx))
                    model_kwargs['past'] = pasts

                if return_dict_in_generate and output_scores:
                    beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

                # increase cur_len
                cur_len = cur_len + 1

                if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                    if not synced_gpus:
                        break
                    else:
                        this_peer_finished = True

            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=stopping_criteria.max_length,
                beam_indices=beam_indices,
            )

            if return_dict_in_generate:
                if not output_scores:
                    sequence_outputs["sequence_scores"] = None

                if self.config.is_encoder_decoder:
                    return BeamSearchEncoderDecoderOutput(
                        sequences=sequence_outputs["sequences"],
                        sequences_scores=sequence_outputs["sequence_scores"],
                        scores=scores,
                        beam_indices=sequence_outputs["beam_indices"],
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                    )
                else:
                    return BeamSearchDecoderOnlyOutput(
                        sequences=sequence_outputs["sequences"],
                        sequences_scores=sequence_outputs["sequence_scores"],
                        scores=scores,
                        beam_indices=sequence_outputs["beam_indices"],
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                    )
            else:
                return sequence_outputs["sequences"]

    def test_epoch_end(self, outputs):
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
        parser.add_argument("--model_path", type=str, help="Path to the checkpoint directory or model name.")
        parser.add_argument("--checkpoints", nargs="+", type=str, help="Checkpoints in model_path to use for ensemble.")
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
        #parser.add_argument("--do_sample", default=False, action="store_true", help='Whether or not to use sampling ; use greedy decoding otherwise.')
        parser.add_argument("--temperature", default=1.0, type=float, help='The value used to module the next token probabilities.')
        parser.add_argument("--top_k", default=50, type=int, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
        parser.add_argument("--top_p", default=1.0, type=float, help='If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are kept for generation.')
        parser.add_argument("--repetition_penalty", default=1.0, type=float, help='The parameter for repetition penalty. 1.0 means no penalty.')
        parser.add_argument("--length_penalty", default=1.0, type=float, help='Exponential penalty to the length. 1.0 means no penalty.')
        parser.add_argument("--output_scores", default=False, action="store_true", help='Whether or not to return the prediction scores.')
        parser.add_argument("--num_return_sequences", default=1, type=int, help='The number of independently computed returned sequences for each element in the batch, i.e. N-best')
        parser.add_argument("--return_dict_in_generate", default=False, action="store_true", help='Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.')
        parser.add_argument("--no_repeat_ngram_size", default=0, type=int, help='If set to int > 0, all ngrams of that size can only occur once.')
        parser.add_argument("--ensemble_mode", default="linear", type=str, help='How to interpolate probabilities from different checkpoints (either "linear" or "log_linear").')


        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")

        return parser


def main(args):

    if Path(args.translation).is_file():
        logging.info("Output file `{}` already exists and will be overwritten...".format(args.translation))
        Path(args.translation).unlink()

    ensemble = Ensemble(args)

    test_set = CustomDatasetForInference(src_file=args.test_source,
                                         tgt_file=args.test_target,
                                         name="test",
                                         tokenizer=ensemble.tokenizer,
                                         max_input_len=args.max_input_len,
                                         max_output_len=args.max_output_len,
                                         src_lang=args.src_lang,
                                         tgt_lang=args.tgt_lang,
                                         src_tags_included=args.src_tags_included,
                                         tgt_tags_included=args.tgt_tags_included,
                                         target_tags=args.target_tags)
    ensemble.set_test_set(test_set)

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None,
                         replace_sampler_ddp=False,
                         limit_test_batches=args.test_percent_check,
                         logger=None,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         precision=32 if args.fp32 else 16, amp_backend='native',
                         )

    trainer.test(ensemble)

    print("Decoded outputs written to {}".format(args.translation))


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="Ensemble")
    parser = Ensemble.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)

