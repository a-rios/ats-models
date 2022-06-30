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
from collections import defaultdict
import sentencepiece.sentencepiece_model_pb2 as pb2

from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def trim_embedding_matrix_of_pretrained_model(
    save_model_to,
    base_model,
    tokenizer,
    cache_dir,
    reduce_to_vocab,
    print_params,
):
    """
    trims embedding matrix based on vocab in `reduce_to_vocab` (optional)
    """
    logger.info("loading pretrained models and config...")
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=base_model, cache_dir=cache_dir)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer, model_max_length=1024, cache_dir=cache_dir)
    config = MBartConfig.from_pretrained(base_model, cache_dir=cache_dir)
    model.config = config

    if print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)

    ## reduce vocabulary of >250k to vocab given in reduce_to_vocab
    ## embedding matrix is model.shared.weight:torch.Size([250027, 1024])
    original_embed_weight = model.model.shared.weight
    original_vocab_size, model_size = original_embed_weight.shape

    # trim embed matrix
    if reduce_to_vocab is not None:
        with open(reduce_to_vocab, 'r') as f:
            keep_pieces = defaultdict()
            for piece in f.readlines():
                # check if this piece is actually in the spm vocab (some junk might not be)
                if tokenizer.sp_model.piece_to_id(piece.rstrip()) > 0:
                    keep_pieces[piece.rstrip()] = 1
                    #print(piece)
                #print(keep_pieces)

            num_special_tokens = 4 # {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
            new_vocab_size = len(keep_pieces) + num_special_tokens + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset + 1 # +1 for mask token at end, +1 fairseq_offset (spm has no <pad>, see https://github.com/ZurichNLP/transformers/blob/df8e6804c004903753d3e635d85f32694e3d2c39/src/transformers/models/mbart/tokenization_mbart.py#L129)
            new_embed_weight = model.model.shared.weight.new_empty(new_vocab_size, model_size)
            ## need to reduce final_logits_bias too
            final_logits_bias_original = model.final_logits_bias.transpose(0,1) # (1, vocab_size)
            final_logits_bias_new = final_logits_bias_original.new_empty(new_vocab_size,1) # TODO: all zeros? not necessary?

            ## keep order same as in original vocab.. iterate over 250k entries
            # `added_vocab_length` = length of special
            # mbart's special tokens used (27)
            added_vocab_length = len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset + 1
            base_vocab_length_original = original_vocab_size - added_vocab_length
            base_vocab_length_new = len(keep_pieces) + num_special_tokens

            ## delete ununsed entries from sentencepiece model of the tokenizer and save the new ModelProto
            pb2_model = pb2.ModelProto()
            pb2_model.ParseFromString(open(os.path.join(cache_dir, "sentencepiece.bpe.model"), 'rb').read())
            indices_to_remove = []
            count=0

            ## from transformers.tokenization_xlm_roberta.py -> self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
            ## sentencepiece model: 0 = <unk>, 1 = <s>, 2 = </s> -> need to copy first 4 rows in embedding matrix and then shift spm ids by +1
            for i in range(0,4):
                piece_embed = original_embed_weight[i]
                piece_final_logits_bias = final_logits_bias_original[i]
                new_embed_weight[i] = piece_embed
                final_logits_bias_new[i] = piece_final_logits_bias

            new_embed_iter = 4
            for embed_iter, spm_iter in zip(range(4,base_vocab_length_original), range(3,base_vocab_length_original-1)): # full vocab size with (!) the added tokens, 250027 |

                if new_embed_iter > base_vocab_length_new:
                    print("ran out of space at position {} in new matrix with vocab size {}".format(j, base_vocab_length_new))
                    exit(0)

                piece = pb2_model.pieces[spm_iter].piece
                #print("embed iter: {}, spm iter {}, piece {}".format(embed_iter, spm_iter, piece))
                if piece in keep_pieces.keys():
                    count +=1
                    ### get embedding
                    piece_embed = original_embed_weight[embed_iter]
                    piece_final_logits_bias = final_logits_bias_original[embed_iter]
                    new_embed_weight[new_embed_iter] = piece_embed
                    final_logits_bias_new[new_embed_iter] = piece_final_logits_bias
                    #print("id : {}, piece {} ".format(new_embed_iter, piece))
                    new_embed_iter +=1
                else:
                    indices_to_remove.append(spm_iter)
                    #print(piece)

            print("total count matched ", count) #
            print("len vocabs to keep {} + special tokens {} + original language tags {} + fairseq_offset {} + <mask> 1 ".format(len(keep_pieces.keys()), num_special_tokens, len(tokenizer.lang_code_to_id), tokenizer.fairseq_offset ))
            print("new vocab size ", new_vocab_size)

            # check ids in reduced spm model
            removed =0
            for i in tqdm(indices_to_remove):
                position = i-removed
                # print("deleting ", pb2_model.pieces[position].piece)
                del pb2_model.pieces[position]
                removed +=1

            ## fill in additional vocab positions (language ids etc)
            for i in range(added_vocab_length):
                new_embed_weight[base_vocab_length_new+i] = original_embed_weight[base_vocab_length_original+i]

                final_logits_bias_new[base_vocab_length_new+i] = final_logits_bias_original[base_vocab_length_original+i]
                #print("position in new tensor ", base_vocab_length_new+i)
                #print("position in old tensor ", base_vocab_length_original+i)
                #print("embed ", new_embed_weight[base_vocab_length_new+i])

            assert len(torch.nonzero(final_logits_bias_new, as_tuple=False)) == 0, "final logits bias must be all zeros for fine-tuning but found non zero values. Hint: check update to new_embed_weights and final_logits_bias_new."



            model.model.shared.weight.data = new_embed_weight
            model.final_logits_bias.data = final_logits_bias_new.transpose(0,1) # swap dimensions back to (1, vocab_size

            with open(os.path.join(save_model_to, 'reduced.spm.model'), 'wb') as f:
                f.write(pb2_model.SerializeToString())

            tokenizer.init_kwargs['vocab_file'] = os.path.join(save_model_to, "reduced.spm.model")
            tokenizer.vocab_file = os.path.join(save_model_to, "reduced.spm.model")
            logger.info(f"saving reduced tokenizer vocabulary with size {new_vocab_size}")
            tokenizer.save_vocabulary(save_model_to)
            config.vocab_size = new_vocab_size

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    logger.info('saving tokenizer')
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def read_in_symbols(infile):
    symbols = set()
    with open(infile, 'r', encoding='utf8') as inf:
        for line in inf:
            line = line.strip().split()
            symbols.update(line)
    return sorted(list(symbols))

def main():
    parser = argparse.ArgumentParser(description="Convert BART to LongBART. Replaces BART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='facebook/mbart-large-cc25',
        help='The name or path of the base model you want to convert'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='facebook/mbart-large-cc25',
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
    parser.add_argument(
        '--add_language_tags',
        type=str, nargs='+',
        help='List of additional language tags (will replace tags given with --replace_tags and initialize with embeddings given with --initialize_tags).'
    )
    parser.add_argument(
        '--initialize_tags',
        type=str, nargs='+',
        help='Initialize new language tags with embeddings of these tags.'
    )
    parser.add_argument("--print-params",
                        action='store_true',
                        help="Print parameter names and shapes.")
    parser.add_argument("--run_tests",
                        action='store_true', help="Run simple dummy tests to check with newly added tags.")

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    if args.add_language_tags is not None:
        assert args.initialize_tags is not None, "Need --initialize_tags to add new language tags"
        assert len(args.add_language_tags) == len(args.initialize_tags), "Need same number of values for --add_language_tags and --initialize_tags but got %i and %i" %(len(args.add_language_tags), len(args.initialize_tags))

    trim_embedding_matrix_of_pretrained_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer=args.tokenizer,
        cache_dir=args.cache_dir,
        reduce_to_vocab=args.reduce_to_vocab,
        print_params=args.print_params,
    )

    tokenizer = MBartTokenizer.from_pretrained(args.save_model_to)
    model = MBartForConditionalGeneration.from_pretrained(args.save_model_to)
    print("loaded tokenizer with len ", len(tokenizer.sp_model))
    print("vocab size in model (+language tags) ", model.config.vocab_size)

    if args.add_language_tags:
        num_added_toks = tokenizer.add_tokens(args.add_language_tags, special_tokens=True)
        print('added', num_added_toks, 'special tokens to tokenizer')
        model.resize_token_embeddings(len(tokenizer))

        for (new_tag, init_tag) in zip(args.add_language_tags, args.initialize_tags):
            init_tag_id = tokenizer.convert_tokens_to_ids(init_tag)
            new_tag_id = tokenizer.convert_tokens_to_ids(new_tag)
            init_embed = model.model.shared.weight[init_tag_id]
            with torch.no_grad():
                model.model.shared.weight[new_tag_id] = init_embed

        tokenizer.add_special_tokens({'additional_special_tokens': args.add_language_tags})
        print("saving tokenizer with new tags")
        tokenizer.save_pretrained(args.save_model_to)
        print("saving model with new tags")
        model.save_pretrained(args.save_model_to)

    if args.run_tests:
        test_tag = args.add_language_tags[0]
        test_tag_id = tokenizer.convert_tokens_to_ids(test_tag)
        print("test tag ", test_tag)
        print("test tag id ", test_tag_id)
        src1 = "de_DE Das ist ein Test."
        src2 = "de_DE Ein zweiter Test."
        trg1 = "Das ist ein einfacher Test."
        trg2 = "Das ist ein zweiter einfacher Test."

        batch: dict = tokenizer.prepare_seq2seq_batch(src_texts=[src1, src2], max_length=1024, truncation=False, padding="max_length", return_tensors="pt")
        print(batch)

        decoder_start_token_ids = [test_tag_id, test_tag_id]
        decoder_start_token_ids = torch.tensor(decoder_start_token_ids)
        print("decoder start ids ", decoder_start_token_ids)
        translated_tokens = model.generate(**batch, decoder_input_ids=decoder_start_token_ids, use_cache=True, num_beams=2)
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(translation)

if __name__ == "__main__":
    main()


