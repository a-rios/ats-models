#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is taken on SimpleFUDGE: https://github.com/ZurichNLP/SimpleFUDGE

Note:
    Authors: Tannon Kew (kew@cl.uzh.ch)

"""
import torch
from torch import Tensor
from transformers import LogitsProcessor
from typing import List, Optional
import json

class FUDGELogits(LogitsProcessor):
    def __init__(self, tokenizer, conditioning_model, condition_lambda, precondition_topk, batch_size, soft, analysis_file=None):
        """
        vocab is a dictionary where the keys are tokens
        and the values are the corresponding ids.
        """
        self.tokenizer = tokenizer
        self.conditioning_model = conditioning_model
        self.condition_lambda = condition_lambda
        self.precondition_topk = precondition_topk
        self.batch_size = batch_size # only used in vectorized implementation
        self.soft = soft
        self.analysis_file = analysis_file

    def __call__(self, input_ids, scores):
        """
        Vectorized implementation of FUDGE as a
        stand-alone function used as a LogitsProcessor. This
        is much fast than the non-vectorized version above.

        :input_ids: shape([num_beams*batch_size, seq_len])
        :scores: shape([num_beams*batch_size, vocab_size])
        """
        # breakpoint()
        num_beams = input_ids.shape[0]//self.batch_size # infer number of beams

        # get precondition logits and indices in vocabulary
        top_logits, top_indices = scores.topk(self.precondition_topk, dim=-1) # scores.shape([num_beams*batch_size, vocab_size])
        # top_logits.shape([num_beams*batch_size, topk])
        # top_indices.shape([num_beams*batch_size, topk])

        ids_expanded = input_ids.repeat_interleave(self.precondition_topk, dim=0) # ids_expanded.shape([topk*num_beams*batch_size, seq_len])

        # NOTE: In original tplus1_candidates is defined
        # as a 3D tensor but gets flattened to 2D when
        # passed to model. We use 2d for simplicity
        tplus1_candidates = torch.cat([ids_expanded, top_indices.view(-1,1)], dim=-1)[:,1:]
        # 2D tplus1_candidates.shape([topk*num_beams*batch_size, seq_len-bos+1])

        cur_len = tplus1_candidates.shape[-1]
        #print("cur len ", cur_len)
        expanded_lengths = torch.LongTensor([[cur_len for _ in range(self.precondition_topk)] for _ in range(self.batch_size)]).to(scores.device)
        #print("exp len ", expanded_lengths.flatten(0, 1))

        # apply conditioning
        condition_logits = self.conditioning_model(
            inputs=tplus1_candidates, # # [batch*topk, seq+1]
            labels=None
            # # lengths=expanded_lengths.flatten(0, 1), # [batch*topk]
            # future_words=None, log_probs=None, syllables_to_go=None, future_word_num_syllables=None, rhyme_group_index=None, run_classifier=False
            )

        # breakpoint()
        condition_logits = condition_logits.view(self.batch_size, self.precondition_topk, -1)[:, :, -1].repeat_interleave(num_beams, dim=0) # shape: [num_beams*batch_size, topk] of last FUDGE pred

        condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs

        fudge_logits = (top_logits + self.condition_lambda * condition_logits)

        if not self.soft: # set all other logits to -inf, i.e. HARD FUDGE
            new_scores = torch.zeros_like(scores).fill_(-float("inf")) # default value for logits = -inf
        else:
            new_scores = scores.clone() # default value for logits = original scores
        # replace original logits with computed fudge
        new_scores.scatter_(1, top_indices, fudge_logits)

        # write logits to file for analysing impact of fudge
        if self.analysis_file is not None:
            with open(self.analysis_file, 'a+', encoding='utf8') as outf:
                d = {
                    'prev_token': self.tokenizer.batch_decode(input_ids[-1])[-1],
                    'time_step': input_ids.shape[-1],
                    'top_tokens': [self.tokenizer.batch_decode(top_indices[i]) for i in range(len(top_indices))],
                    'pre_scores': top_logits.tolist(),
                    'post_scores': fudge_logits.tolist(),
                }
                # breakpoint()
                d = json.dumps(d)
                outf.write(f'{d}\n')

        return new_scores

if __name__ == "__main__":
    pass
