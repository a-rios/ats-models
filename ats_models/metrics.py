#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is adapted from AllenAI's Longformer:
    https://github.com/allenai/longformer/

Note:
    Authors: Annette Rios (arios@cl.uzh.ch) Tannon Kew (kew@cl.uzh.ch)

"""

import torch
from rouge_score import rouge_scorer
import sacrebleu

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def get_eval_scores(gold_strs, generated_strs, vloss=None):
    if vloss is None:
        vloss = torch.zeros(len(gold_strs))

    scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
    rouge1 = rouge2 = rougel = rougelsum = 0.0
    for ref, pred in zip(gold_strs, generated_strs):
        score = scorer.score(ref, pred)
        rouge1 += score['rouge1'].fmeasure
        rouge2 += score['rouge2'].fmeasure
        rougel += score['rougeL'].fmeasure
        rougelsum += score['rougeLsum'].fmeasure
    rouge1 /= len(generated_strs)
    rouge2 /= len(generated_strs)
    rougel /= len(generated_strs)
    rougelsum /= len(generated_strs)
    if gold_strs is None or gold_strs == "":
        print("gold ", gold_strs)
        print("generated ", generated_strs)
    bleu = sacrebleu.corpus_bleu(generated_strs, [gold_strs])

    return {'vloss': vloss,
            'rouge1': vloss.new_zeros(1) + rouge1,
            'rouge2': vloss.new_zeros(1) + rouge2,
            'rougeL': vloss.new_zeros(1) + rougel,
            'rougeLsum': vloss.new_zeros(1) + rougelsum,
            'bleu' : vloss.new_zeros(1) + bleu.score,
            'decoded' : generated_strs}
