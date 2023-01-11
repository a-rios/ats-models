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
import sentencepiece.sentencepiece_model_pb2 as pb2
import sentencepiece as spm

from transformers import MBartTokenizer, MBartForConditionalGeneration


from .long_models.sliding_chunks import pad_to_window_size
from .long_models.longformer_mbart import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig, LongformerSelfAttentionForMBart

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
    reduce_to_vocab,
    print_params
):
    logger.info("loading pretrained models and config...")
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=base_model, cache_dir=cache_dir)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos, cache_dir=cache_dir)
    tokenizer.save_vocabulary(cache_dir)
    config = MLongformerEncoderDecoderConfig.from_pretrained(base_model, cache_dir=cache_dir)
    print(config)

    if print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = ['MLongformerEncoderDecoderForConditionalGeneration', ]

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    assert current_max_pos == config.max_position_embeddings + 2

    config.max_encoder_position_embeddings = max_pos
    config.max_decoder_position_embeddings = config.max_position_embeddings
    del config.max_position_embeddings ## will be filled in from_pretrained with default value 1024, will initialize model.encoder.embed_positions.weight as (1026, 1024) instead of max_encoder_position_embeddings --> changed in line 679 of transformers.models.mbart.modeling_mbart for encoder, line 778 for decoder, also changed init in lines 640 (encoder) and 787 (decoder), as max length is again read from config instead of using defined values :/
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
        longformer_self_attn_for_bart = LongformerSelfAttentionForMBart(config, layer_id=i)

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
    parser = argparse.ArgumentParser(description="Convert mBART to LongmBART. Replaces mBART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='facebook/mbart-large-cc25',
        help='The name or path of the base model you want to convert'
    )
    parser.add_argument(
        '--tokenizer_name_or_path',
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
    parser.add_argument(
        '--add_to_vocab',
        type=str, nargs='+',
        help='List of additional tokens to be added to the vocab.'
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

    create_long_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        attention_window=args.attention_window,
        max_pos=args.max_pos,
        cache_dir=args.cache_dir,
        reduce_to_vocab=args.reduce_to_vocab,
        print_params=args.print_params
    )
    tokenizer = MBartTokenizer.from_pretrained(args.save_model_to)
    config = MLongformerEncoderDecoderConfig.from_pretrained(args.save_model_to)
    #print(config)
    model = MLongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.save_model_to)
    print(model)
    print("loaded tokenizer with len ", len(tokenizer.sp_model))

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

    if args.add_to_vocab is not None:
        print("adding new tokens to vocab")
        print("provided {} tokens".format(len(args.add_to_vocab)))
        new_tokens = set(args.add_to_vocab) - set(tokenizer.get_vocab().keys())
        print("number of tokens not yet part of vocab: {}".format(len(new_tokens)))
        tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(tokenizer))

    if args.add_language_tags is not None or args.add_to_vocab:
        print("saving tokenizer with new tags/vocab")
        tokenizer.save_pretrained(args.save_model_to)
        print("saving model with new tags/vocab")
        model.save_pretrained(args.save_model_to)

    print("special tokens map ", tokenizer.special_tokens_map)
    print("id-to-lang-code ",tokenizer.id_to_lang_code)
    print("lang-code-to-id", tokenizer.lang_code_to_id)

    ## check embeddings
    if args.add_language_tags is not None and args.initialize_tags is not None:
        for new_tag, init_tag in zip(args.add_language_tags, args.initialize_tags):
            print("original language embedding for {}: {}".format(init_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(init_tag)]))
            print("initialized {} with embedding: {}".format(new_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(new_tag)]))

    if args.run_tests:
        test_tag = args.add_language_tags[0]
        test_tag_id = tokenizer.convert_tokens_to_ids(test_tag)
        src_tag_id = tokenizer.convert_tokens_to_ids('de_DE')
        print("test tag ", test_tag)
        print("test tag id ", test_tag_id)
        print("src test id ", src_tag_id)
        src1 = "Das ist ein Test."
        src2 = "Ein zweiter Test."
        trg1 = "Das ist ein einfacher Test."
        trg2 = "Das ist ein zweiter einfacher Test."

        tokenizer.src_lang="de_DE"
        batch: dict = tokenizer(text=[src1, src2], max_length=1024,  truncation=False, padding="max_length", return_tensors="pt")

        # custom attention mask for longformer attention:
        long_att_mask = batch['attention_mask']
        long_att_mask[ batch['input_ids'] == src_tag_id] = 2
        batch['attention_mask'] = long_att_mask
        print(batch)

        decoder_start_token_ids = [test_tag_id, test_tag_id]
        decoder_start_token_ids = torch.tensor(decoder_start_token_ids).unsqueeze(1)
        print("decoder start ids ", decoder_start_token_ids)
        translated_tokens = model.generate(**batch, decoder_input_ids=decoder_start_token_ids, use_cache=True, num_beams=2)
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(translation)

if __name__ == "__main__":
    main()


