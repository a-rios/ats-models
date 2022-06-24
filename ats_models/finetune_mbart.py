#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is adapted from AllenAI's Longformer
summarization task:
    https://github.com/allenai/longformer/blob/master/scripts/summarization.py

Note:
    Annette Rios (arios@cl.uzh.ch) initially adapted it for long-document simplication.
    Tannon Kew (kew@cl.uzh.ch) made minor changes for its
    application in the ReAdvisor project for response
    generation.

Date: 04/06/2021

"""

import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from rouge_score import rouge_scorer
import sacrebleu

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.plugins import DDPPlugin

import logging
from transformers import MBartTokenizer, MBartForConditionalGeneration, MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right
import datasets
from functools import partial
from .data import CustomDataset


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

def get_eval_scores(gold_strs, generated_strs, remove_tgt_tag=False, vloss=None):
    if vloss is None:
        vloss = torch.zeros(len(gold_strs))
    if remove_tgt_tag: # not needed with skip_special_tokens?
        # remove tags from target text
        # print(gold_strs)
        gold_strs = [' '.join(r.split(' ')[1:]) for r in gold_strs]
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


class FineTuner(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params

        if self.args.from_pretrained is not None: ## TODO check how to do this with resume_ckpt
            self._set_config()
            self._load_pretrained()

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        self.current_checkpoint =0
        self.best_checkpoint = None
        self.best_metric = 10000 if self.args.early_stopping_metric == 'vloss' else 0 ## keep track of best dev value of whatever metric is used in early stopping callback
        self.num_not_improved = 0
        self.save_hyperparameters()

    def _load_pretrained(self):
        self.model = MBartForConditionalGeneration.from_pretrained(self.args.from_pretrained, config=self.config)
        self.tokenizer = MBartTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)
        if self.args.tags_included:
            self.tokenizer = MBartTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)
        elif self.args.src_lang is not None and self.args.tgt_lang is not None:
            self.tokenizer = MBartTokenizer.from_pretrained(self.args.tokenizer, use_fast=True, src_lang=self.args.src_lang, tgt_lang=self.args.tgt_lang)
        else:
            print(f"Either tags must be included as first token for each sample in input and output (--tags_included), or set for the entire corpus as --src_lang and --tgt_lang.")

    def _set_config(self):
        self.config = MBartConfig.from_pretrained(self.args.from_pretrained)
        self.config.attention_dropout = self.args.attention_dropout
        self.config.dropout = self.args.dropout
        self.config.activation_dropout = self.args.activation_dropout
        self.config.gradient_checkpointing = self.args.grad_ckpt

    def forward(self, input_ids, output_ids):
        input_ids, attention_mask = CustomDataset.prepare_input(input_ids, self.tokenizer.pad_token_id)
        decoder_input_ids = shift_tokens_right(output_ids, self.config.pad_token_id) # (in: output_ids, eos_token_id, tgt_lang_id out: tgt_lang_id, output_ids, eos_token_id)
        labels = decoder_input_ids[:, 1:].clone()
        decoder_input_ids = decoder_input_ids[:, :-1] # without eos/last pad
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)

        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,)
        lm_logits = outputs[0]
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.tokenizer.pad_token_id
            )
        return [loss]

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        self.log('train-loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        input_ids, output_ids = batch
        input_ids, attention_mask = CustomDataset.prepare_input(input_ids, self.tokenizer.pad_token_id)
        if self.args.tags_included:
            # get list of target language tags
            # output_ids (batch_size, seq_len), with padding
            shifted = shift_tokens_right(output_ids, self.config.pad_token_id) # (in: output_ids, eos_token_id, tgt_lang_id out: tgt_lang_id, output_ids, eos_token_id)
            decoder_start_token_ids = shifted.narrow(dim=1, start=0, length=1)
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_ids=decoder_start_token_ids)
        else:
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang])

        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)

        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # get scores as dict
        scores = get_eval_scores(gold_str, generated_str, False, vloss) # with skip_special_tokens=True, no language tag

        outfile = self.args.save_dir + "/" + args.save_prefix + "/_val_out_checkpoint_" + str(self.current_checkpoint)

        with open(outfile, 'a') as f:
            for sample in generated_str:
                f.write(sample + "\n")
        self.log('vloss', vloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('bleu', scores['bleu'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('rouge1', scores['rouge1'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('rouge2', scores['rouge2'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('rougeL', scores['rougeL'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('rougeLsum', scores['rougeLsum'], on_step=False, on_epoch=True, prog_bar=True)

        return scores

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
            metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        print("Evaluation on checkpoint [{}] ".format(self.current_checkpoint))
        print(logs)

        ## save metric value + number of checkpoint if best
        if self.args.early_stopping_metric == 'vloss' and logs['vloss'] < self.best_metric:
            self.best_metric = logs['vloss']
            self.best_checkpoint = self.current_checkpoint
            print("New best checkpoint {}, with {} {}.".format(self.best_checkpoint, self.best_metric, self.args.early_stopping_metric))
        elif logs[self.args.early_stopping_metric] > self.best_metric:
            self.best_metric = logs[self.args.early_stopping_metric]
            self.best_checkpoint = self.current_checkpoint
            print("New best checkpoint {}, with {} {}.".format(self.best_checkpoint, self.best_metric, self.args.early_stopping_metric))
        self.current_checkpoint +=1


    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        print(result)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode=self.lr_mode, factor=self.args.lr_reduce_factor, patience=self.args.lr_reduce_patience)
        return {
       'optimizer': self.optimizer,
       'lr_scheduler': self.scheduler,
       'monitor': self.args.early_stopping_metric
        }

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader

        dataset = CustomDataset(inputs=self.datasets[split_name + "_source"], labels=self.datasets[split_name + "_target"] , name=split_name, tokenizer=self.tokenizer,
                                       max_input_len=self.args.max_input_len, max_output_len=self.args.max_output_len, src_lang=self.args.src_lang, tgt_lang=self.args.tgt_lang, tags_included=self.args.tags_included)

        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)

        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=partial(CustomDataset.collate_fn, pad_token_id=self.tokenizer.pad_token_id))

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'val', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
        parser.add_argument("--save_dir", type=str, default='simplification', help="Directory to save models.")
        parser.add_argument("--save_prefix", type=str, default='test', help="subfolder in save_dir for this model")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,  help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument("--num_sanity_val_steps", type=int, default=0,  help="Number of evaluation sanity steps to run before starting the training. Default: 0.")

        #data
        parser.add_argument("--train_source", type=str, default=None,  help="Path to the source train file.")
        parser.add_argument("--train_target", type=str, default=None, help="Path to the target train file.")
        parser.add_argument("--val_source", type=str, default=None, help="Path to the source validation file.")
        parser.add_argument("--val_target", type=str, default=None, help="Path to the target validation file.")
        parser.add_argument("--test_source", type=str, default=None, help="Path to the source test file (to evaluate after training is finished).")
        parser.add_argument("--test_target", type=str, default=None, help="Path to the target test file (to evaluate after training is finished).")
        parser.add_argument("--src_lang", type=str, default=None, help="Source language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tgt_lang", type=str, default=None, help="Target language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tags_included", action='store_true', help="Text files already contain special tokens (language tags and </s>. Source:  src_tag seq, Target:  tgt_tag seq. Note: actual source sequence is seq src_tag </s>, will be changed internally after possibly clipping sequence to given max_length.")
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=512, help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--wandb", type=str, default=None, help="WandB project name to use if logging fine-tuning with WandB.")
        parser.add_argument("--wandb_entity", type=str, default=None, help="WandB account name to use if logging fine-tuning with WandB.")

        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulation steps.")
        parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator for pytorch lightning trainer (gpu or cpu).")
        parser.add_argument("--devices", type=int, nargs="+", required=True, help="Device id(s).")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")

        ## model params:
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
        parser.add_argument("--activation_dropout", type=float, default=0.0, help="activation_dropout")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum change in the monitored quantity to qualify as an improvement.")

        # Optimization params:
        #parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Initial learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations in percent of an epoch.")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--train_percent_check", default=1.00, type=float, help='Percent of training data used (for testing) NOTE: not available in pytprch lightning==1.1.6')
        parser.add_argument("--max_epochs", type=int, default=100000, help="Maximum number of epochs (will stop training even if patience for early stopping has not been reached).")
        parser.add_argument("--early_stopping_metric", type=str, default='rougeL', help="Metric to be used for early stopping: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu")
        parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
        parser.add_argument("--lr_reduce_patience", type=int, default=8, help="Patience for LR reduction in Plateau scheduler.")
        parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Learning rate reduce factor for Plateau scheduler.")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--save_top_k", type=int, default=5, help="Number of best checkpoints to keep. Others will be removed.")
        parser.add_argument("--save_every_n_val_epochs", type=int, default=0, help="Number of validation epochs between checkpoints.")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')

        ## inference params
        parser.add_argument("--decoded", type=str, default='decoded.out', help="Output file to write decoded sequence to.")
        parser.add_argument("--beam_size", type=int, default=4, help="Beam size for inference when testing/validating. Default: 4.")
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')

        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = FineTuner(args)

    if args.print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)


    model.datasets = datasets.load_dataset('text', data_files={'train_source': args.train_source, 'train_target': args.train_target, 'val_source': args.val_source, 'val_target': args.val_target, 'test_source': args.test_source, 'test_target': args.test_target })

    if args.wandb:
        logger = WandbLogger(project=args.wandb, entity=args.wandb_entity)
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(args.save_dir, args.save_prefix), name="tensorboard_logs")

    print(args)

    model.lr_mode='max'
    if args.early_stopping_metric == 'vloss':
        model.lr_mode='min'
    early_stop_callback = EarlyStopping(monitor=args.early_stopping_metric, min_delta=args.min_delta, patience=args.patience, verbose=True, mode=model.lr_mode) # metrics: val_loss, bleu, rougeL

    checkpoint_name = "checkpoint{{epoch:02d}}_{{{}".format(args.early_stopping_metric)
    checkpoint_name += ':.5f}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix),
        filename=checkpoint_name,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.early_stopping_metric,
        mode=model.lr_mode)

    progress_bar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_epochs=args.max_epochs if not args.debug else 100,
                         max_steps=None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=args.num_sanity_val_steps,
                         check_val_every_n_epoch=1 if not (args.debug) else 1,
                         limit_val_batches=args.val_percent_check,
                         limit_test_batches=args.test_percent_check,
                         logger=logger,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         precision=32 if args.fp32 else 16, amp_backend='native',
                         resume_from_checkpoint=args.resume_ckpt,
                         callbacks=[early_stop_callback, checkpoint_callback, progress_bar_callback]
                         )
    ## write config + tokenizer to save_dir
    model.model.save_pretrained(args.save_dir + "/" + args.save_prefix)
    model.tokenizer.save_pretrained(args.save_dir + "/" + args.save_prefix)
    trainer.fit(model)
    print("Training ended. Best checkpoint {} with {} {}.".format(model.best_checkpoint, model.best_metric, args.early_stopping_metric))
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="Fine-tune MBart")
    parser = FineTuner.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)

