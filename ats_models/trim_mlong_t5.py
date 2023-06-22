#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is adapted from AllenAI's Longformer:
    https://github.com/allenai/longformer/

Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

"""
import argparse
import logging
import os
import copy
from collections import defaultdict
import sentencepiece.sentencepiece_model_pb2 as pb2
import sentencepiece as spm

from transformers import T5Tokenizer, LongT5ForConditionalGeneration,  LongT5Config
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_reduced_vocab_model(
    save_model_to,
    base_model,
    tokenizer,
    cache_dir,
    reduce_to_vocab,
    print_params
):

    config = LongT5Config.from_pretrained(base_model, cache_dir=cache_dir)
    model = LongT5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=base_model, config=config, cache_dir=cache_dir)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer,  cache_dir=cache_dir)
    tokenizer.save_pretrained(cache_dir)

    if print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)

    ## reduce vocabulary of >250k to vocab given in reduce_to_vocab
    ## len(tokenizer)/sp_model is 256384
    ## embedding matrix is shared.weight:torch.Size([256384, 768])
    ## also reduce lm_head.weight:torch.Size([256384, 768])
    ## tokenizer len is 256300. Indices 17-272 are hex (<0xBC>), 4-16 are prefixes ([translate]), 0: <pad>, 1: </s>, 2: <s>, 3: <unk>
    original_embed_weight = model.shared.weight
    original_lm_head = model.lm_head.weight
    original_vocab_size, model_size = original_embed_weight.shape

    if reduce_to_vocab is not None:
        with open(reduce_to_vocab, 'r') as f:
            keep_pieces = defaultdict()
            for piece in f.readlines():
                # check if this piece is actually in the spm vocab (some junk might not be)
                if tokenizer.sp_model.piece_to_id(piece.rstrip()) > 0:
                    keep_pieces[piece.rstrip()] = 1
                    #print(piece)
                #print(len(keep_pieces))

            ## embedding matrix has 84 extra positions: sp_model = 256300, embed matrix = 256384
            num_additional_tokens_in_embed=84
            num_special_tokens = 273 # <pad>, </s>, <unk> + 256 <hex>
            num_extra_ids = 100 # last 100 entires are extra_ids used for spans in pre-training, delete
            new_vocab_size = len(keep_pieces) + num_special_tokens
            new_embed_weight = model.shared.weight.new_empty(new_vocab_size, model_size)
            ## need to reduce lm_head too
            lm_head_new = original_lm_head.new_empty(new_vocab_size,model_size)

            ## keep order same as in original vocab.. iterate over 250k entries
            ## sp_model = 256300, embed matrix = 256384
            base_vocab_length_original = original_vocab_size - num_extra_ids - num_additional_tokens_in_embed
            base_vocab_length_new = len(keep_pieces) + num_special_tokens

            print("new vocab size ", new_vocab_size)
            print("base vocab len in orig ", base_vocab_length_original)
            print("new vocab len ", base_vocab_length_new)

            ## delete ununsed entries from sentencepiece model of the tokenizer and save the new ModelProto
            pb2_model = pb2.ModelProto()
            pb2_model.ParseFromString(open(os.path.join(cache_dir, "spiece.model"), 'rb').read())
            indices_to_remove = []
            count=0

            ## keep piece 0,1, and 2 -> these are <pad>, </s>, <unk> + 256 <hex> codes
            for i in range(0,num_special_tokens):
                piece_embed = original_embed_weight[i]
                piece_lm_head = original_lm_head[i]
                new_embed_weight[i] = piece_embed
                lm_head_new[i] = piece_lm_head

            new_embed_iter = num_special_tokens
            test_matched = []
            for embed_iter, spm_iter in zip(range(num_special_tokens,base_vocab_length_original), range(num_special_tokens,base_vocab_length_original)):

                if new_embed_iter > base_vocab_length_new:
                    print("ran out of space at position {} in new matrix with vocab size {}".format(j, base_vocab_length_new))
                    exit(0)

                piece = pb2_model.pieces[spm_iter].piece
                # print("embed iter: {}, spm iter {}, piece {}".format(embed_iter, spm_iter, piece))
                if piece in keep_pieces.keys():
                    count +=1
                    ### get embedding
                    piece_embed = original_embed_weight[embed_iter]
                    piece_lm_head = original_lm_head[embed_iter]
                    new_embed_weight[new_embed_iter] = piece_embed
                    lm_head_new[new_embed_iter] = piece_lm_head
                    #print("keep id : {}, piece {} ".format(new_embed_iter, piece))
                    new_embed_iter +=1
                    test_matched.append(piece)
                else:
                    indices_to_remove.append(spm_iter)
                    #print(piece)

            print("total count matched ", count) #
            print("len vocabs to keep {} + special tokens {}".format(len(keep_pieces.keys()), num_special_tokens ))
            print("new vocab size ", new_vocab_size)

            #for keep in keep_pieces.keys():
                #if not keep in test_matched:
                    #print("piece {} in keep_pieces but not matched in spm ".format(keep))
            #exit(0)

            # check ids in reduced spm model
            removed =0
            # print("indices to remove ", indices_to_remove)
            for i in indices_to_remove:
                position = i-removed
                #print("deleting ", pb2_model.pieces[position].piece)
                del pb2_model.pieces[position]
                removed +=1


            ## fill in the 100 extra ids at the end TODO do we even need those?
            # for i in range(num_extra_ids):
            #     new_embed_weight[base_vocab_length_new+i] = original_embed_weight[base_vocab_length_original+i]
            #     lm_head_new[base_vocab_length_new+i] = original_lm_head[base_vocab_length_original+i]
            #     #print("position in new tensor ", base_vocab_length_new+i)
            #     #print("position in old tensor ", base_vocab_length_original+i)
            #     #print("embed ", new_embed_weight[base_vocab_length_new+i])

            model.shared.weight.data = new_embed_weight
            model.lm_head.weight.data = lm_head_new

            with open(os.path.join(save_model_to, 'reduced.spm.model'), 'wb') as f:
                f.write(pb2_model.SerializeToString())

            tokenizer.init_kwargs['vocab_file'] = os.path.join(save_model_to, "reduced.spm.model")
            tokenizer.vocab_file = os.path.join(save_model_to, "reduced.spm.model")
            model.config.vocab_size = new_vocab_size
            tokenizer.save_vocabulary(save_model_to)
            print("saving tokenizer with len ", len(tokenizer.sp_model))
            #tokenizer.save_pretrained(save_model_to)

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    print("saving tokenizer")
    tokenizer.save_pretrained(save_model_to)
    #print(model)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Trim mlong-t5's vocabulary to the given list of items.")
    # agemagician/mlong-t5-tglobal-base, agemagician/mlong-t5-tglobal-large, agemagician/mlong-t5-tglobal-xl
    parser.add_argument(
        '--base_model',
        type=str,
        default='agemagician/mlong-t5-tglobal-base',
        help='The name or path of the base model you want to convert'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='agemagician/mlong-t5-tglobal-base',
        help='The name or path of the tokenizer'
    )
    parser.add_argument(
        '--save_model_to',
        type=str,
        required=True,
        help='The path to save the converted model'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        help='where to save original model'
    )
    parser.add_argument(
        '--reduce_to_vocab',
        type=str,
        help='List of subword entries to keep in new model (one token per line).'
    )
    parser.add_argument("--print-params",
                        action='store_true',
                        help="Print parameter names and shapes.")

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    create_reduced_vocab_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer=args.tokenizer,
        cache_dir=args.cache_dir,
        reduce_to_vocab=args.reduce_to_vocab,
        print_params=args.print_params
    )
    # check if new model + tokenizer can be loaded
    tokenizer = T5Tokenizer.from_pretrained(args.save_model_to)
    model = LongT5ForConditionalGeneration.from_pretrained(args.save_model_to)
    print("loaded tokenizer with len ", len(tokenizer.sp_model))


if __name__ == "__main__":
    main()


