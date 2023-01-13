#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is adapted from AllenAI's Longformer:
    https://github.com/allenai/longformer/

Note:
    Authors: Annette Rios (arios@cl.uzh.ch) Tannon Kew (kew@cl.uzh.ch)

"""
import argparse
import logging
import os
from tqdm import tqdm
import copy
from collections import defaultdict

from transformers import BartTokenizer, BartForConditionalGeneration


from .long_models.sliding_chunks import pad_to_window_size
from .long_models.longformer_bart import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig, LongformerSelfAttentionForBart

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_long_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    attention_window,
    max_pos,
    cache_dir,
    print_params
):
    logger.info("loading pretrained models and config...")
    model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=base_model, cache_dir=cache_dir)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos, cache_dir=cache_dir)
    tokenizer.save_vocabulary(cache_dir)
    config = LongformerEncoderDecoderConfig.from_pretrained(base_model, cache_dir=cache_dir)
    print(config)

    if print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = ['LongformerEncoderDecoderForConditionalGeneration', ]

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    assert current_max_pos == config.max_position_embeddings + 2

    config.max_encoder_position_embeddings = max_pos
    config.max_decoder_position_embeddings = config.max_position_embeddings
    del config.max_position_embeddings
    max_pos += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    assert max_pos >= current_max_pos

    # allocate a larger position embedding matrix for the encoder
    new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed
    model.config = config
    print("config ", config)

    # allocate a larger position embedding matrix for the decoder
    # new_decoder_pos_embed = model.model.decoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # # copy position embeddings over and over to initialize the new position embeddings
    # k = 2
    # step = current_max_pos - 2
    # while k < max_pos - 1:
    #     new_decoder_pos_embed[k:(k + step)] = model.model.decoder.embed_positions.weight[2:]
    #     k += step
    # model.model.decoder.embed_positions.weight.data = new_decoder_pos_embed

    # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_bart = LongformerSelfAttentionForBart(config, layer_id=i)

        longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_bart.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
        longformer_self_attn_for_bart.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
        longformer_self_attn_for_bart.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

        longformer_self_attn_for_bart.output = layer.self_attn.out_proj

        layer.self_attn = longformer_self_attn_for_bart

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    logger.info('saving tokenizer')
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert BART to LongBART. Replaces BART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='facebook/bart-base',
        help='The name or path of the base model you want to convert (facebook/bart-base or facebook/bart-large)'
    )
    parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default='facebook/bart-base',
        help='The name or path of the tokenizer '
    )
    parser.add_argument(
        '--save_model_to',
        type=str,
        required=True,
        help='The path to save the converted model'
    )
    parser.add_argument(
        '--attention_window',
        type=int,
        default=512,
        help='attention window size for longformer self attention (one sided)'
    )
    parser.add_argument(
        '--max_pos',
        type=int,
        default=4096,
        help='maximum encoder positions'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        required=True,
        help='where to save original model'
    )
    parser.add_argument("--print-params",
                        action='store_true',
                        help="Print parameter names and shapes.")

    parser.add_argument("--run_tests",
                        action='store_true', help="Run simple dummy tests to check with newly added tags.")

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    create_long_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        attention_window=args.attention_window,
        max_pos=args.max_pos,
        cache_dir=args.cache_dir,
        print_params=args.print_params
    )
    tokenizer = BartTokenizer.from_pretrained(args.save_model_to)
    config = LongformerEncoderDecoderConfig.from_pretrained(args.save_model_to)
    #print(config)
    model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.save_model_to)
    print(model)

    if args.run_tests:
        src1 = "This is a test."
        src2 = "This is another test."

        batch: dict = tokenizer(text=[src1, src2], max_length=1024,  truncation=False, padding="max_length", return_tensors="pt")
        print(batch)

        print("bos ", tokenizer.bos_token_id)
        print("eos ", tokenizer.eos_token_id)
        print("pad ", tokenizer.pad_token_id)

        # custom attention mask for longformer attention:
        long_att_mask = batch['attention_mask']
        long_att_mask[ batch['input_ids'] == tokenizer.eos_token_id] = 2
        long_att_mask[ batch['input_ids'] == tokenizer.bos_token_id] = 2
        batch['attention_mask'] = long_att_mask
        print(batch)


        translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.bos_token_id, use_cache=True, num_beams=2)
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(translation)

if __name__ == "__main__":
    main()


