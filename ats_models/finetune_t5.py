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
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from rouge_score import rouge_scorer
import sacrebleu
import re

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

import logging
# from transformers import T5Tokenizer, LongT5ForConditionalGeneration,  LongT5Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import datasets
from typing import Optional, Union
from functools import partial
from .t5_data import T5Dataset
from .metrics import label_smoothed_nll_loss, get_eval_scores
from .finetune_mbart import LitProgressBar, remove_special_tokens

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, LoraModel, PeftModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class T5Trainer(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params

        if self.args.lora:
            self.lora_config = LoraConfig(peft_type="LORA",
                                        task_type="SEQ_2_SEQ_LM",
                                        r=self.args.lora_r, lora_alpha=self.args.lora_alpha,
                                        target_modules=self.args.lora_targets,
                                        lora_dropout=self.args.lora_dropout)

        if self.args.from_pretrained is not None:
            self._set_config()
            self._load_pretrained()

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        self.best_checkpoint = None
        self.best_metric = 10000 if self.args.early_stopping_metric == 'vloss' else 0 ## keep track of best dev value of whatever metric is used in early stopping callback
        self.num_not_improved = 0
        self.save_hyperparameters()
        self.validation_step_outputs = []

    def _load_pretrained(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.args.from_pretrained, config=self.config)
        if self.args.lora: # doesn't work (yet) LongT5, see here: https://github.com/huggingface/peft/issues/522
            model.enable_input_require_grads()
            self.model = get_peft_model(model, self.lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = model

    def _set_config(self):
        self.config = AutoConfig.from_pretrained(self.args.from_pretrained)
        self.config.attention_dropout = self.args.attention_dropout
        self.config.dropout = self.args.dropout
        self.config.activation_dropout = self.args.activation_dropout
        self.config.gradient_checkpointing = self.args.grad_ckpt


    def forward(self, input_ids, labels):
        input_ids, attention_mask = T5Dataset.get_attention_mask(input_ids, self.tokenizer.pad_token_id)

        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=None, # decoder_input_ids are created in the LongT5ForConditionalGeneration's forward function from labels
                labels=labels,
                use_cache=False,)
        lm_logits = outputs['logits']
        if self.args.label_smoothing == 0:
            loss = outputs['loss']
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
        self.log('train-loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        input_ids, labels = batch
        input_ids, attention_mask = T5Dataset.get_attention_mask(input_ids, self.tokenizer.pad_token_id)

        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.pad_token_id) # bos in T5 is pad

        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

        gold_str = self.tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # get scores as dict
        scores = get_eval_scores(gold_str, generated_str, vloss=vloss) # with skip_special_tokens=True, no language tag

        outfile = self.args.save_dir + "/" + args.save_prefix + "/_val_out_checkpoint_" + str(self.current_checkpoint)

        with open(outfile, 'a') as f:
            for sample in generated_str:
                f.write(sample + "\n")
        self.log('vloss', vloss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('bleu', scores['bleu'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('rouge1', scores['rouge1'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('rouge2', scores['rouge2'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('rougeL', scores['rougeL'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('rougeLsum', scores['rougeLsum'], on_step=False, on_epoch=True, prog_bar=False)

        self.validation_step_outputs.append(scores)

        return scores

    def on_validation_epoch_end(self):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in self.validation_step_outputs]).mean()
            torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
            metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        print("\nEvaluation on checkpoint [{}] ".format(self.current_checkpoint))
        for m,v in logs.items():
            print(f"{m}:{v}")
        self.current_checkpoint +=1
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=self.lr_mode,
            factor=self.args.lr_reduce_factor,
            patience=self.args.lr_reduce_patience,
            cooldown=self.args.lr_cooldown,
            verbose=True)

        interval = "step" if args.val_check_interval is not None else "epoch"
        frequency = args.val_check_interval if args.val_check_interval is not None else args.check_val_every_n_epoch

        if self.args.lr_scheduler == "plateau":
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": interval,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": frequency,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": self.args.early_stopping_metric,
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "plateauLR",
            }
        elif self.args.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.cosine_Tmax, verbose=True)
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": interval,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                # "frequency": frequency,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "CosineAnnealingLR",
            }
        elif self.args.lr_scheduler == "cosine_wr": # cosine annealing with warm restarts
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.args.cosine_T_0, T_mult=self.args.cosine_T_mult, verbose=True)
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": interval,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                # "frequency": frequency,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "CosineAnnealingWarmRestarts",
            }
        else:
            print(f"Unsupported learning rate scheduler: {self.args.lr_scheduler}. Use one of 'cosine', 'multistep', 'plateau', 'cyclic'.")
            exit(1)
        return [self.optimizer], [lr_scheduler_config]

    def set_datasets(self, train_set: T5Dataset, dev_set: T5Dataset, test_set: Optional[T5Dataset]):
        self.train_set = train_set
        self.dev_set = dev_set
        if test_set is not None:
            self.test_set = test_set

    def set_test_set(self, test_set: T5Dataset):
        self.test_set = test_set

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader

        if split_name == "train":
            dataset = self.train_set
        elif split_name == "dev":
            dataset = self.dev_set
        elif split_name == "test":
            dataset = self.test_set
        else:
            self.log(f"Invalid split name: {split_name}")

        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)

        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=partial(T5Dataset.collate_fn, pad_token_id=self.tokenizer.pad_token_id))

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'dev', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['ckpt_idx'] = self.current_checkpoint

    def on_load_checkpoint(self, checkpoint) -> None:
        self.current_checkpoint = checkpoint['ckpt_idx']

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
        parser.add_argument("--save_dir", type=str, default='simplification', help="Directory to save models.")
        parser.add_argument("--save_prefix", type=str, default='test', help="subfolder in save_dir for this model")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Continue fine-tuning a trained checkpoint but start training from scratch, i.e. parameters, but not optimizer/lr schedulers etc.")
        parser.add_argument("--from_pretrained", type=str, default=None,  help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument("--num_sanity_val_steps", type=int, default=0,  help="Number of evaluation sanity steps to run before starting the training. Default: 0.")

        # lora
        parser.add_argument("--lora", action="store_true",  help="Use LoRa for fine-tuning.")
        parser.add_argument("--lora_targets", type=str, nargs='+', default=["k", "q", "v"], help="Parameters to fine-tune with LoRa. Default: ['k', 'q', 'v']")
        parser.add_argument("--lora_alpha", type=int,  default=32, help="LoRa alpha")
        parser.add_argument("--lora_r", type=int,  default=8, help="LoRa r")
        parser.add_argument("--lora_dropout", type=float,  default=0.01, help="LoRa dropout")

        #data
        parser.add_argument("--train_source", type=str, default=None,  help="Path to the source train file.")
        parser.add_argument("--train_target", type=str, default=None, help="Path to the target train file.")
        parser.add_argument("--dev_source", type=str, default=None, help="Path to the source validation file.")
        parser.add_argument("--dev_target", type=str, default=None, help="Path to the target validation file.")
        parser.add_argument("--test_source", type=str, default=None, help="Path to the source test file (to evaluate after training is finished).")
        parser.add_argument("--test_target", type=str, default=None, help="Path to the target test file (to evaluate after training is finished).")
        parser.add_argument("--train_jsons", type=str, nargs='+', default=None,  help="Path to UZH json file(s) with training data.")
        parser.add_argument("--dev_jsons", type=str, nargs='+', default=None,  help="Path to UZH json file(s) with dev data.")
        parser.add_argument("--test_jsons", type=str, nargs='+', default=None,  help="Path to UZH json file(s) with test data.")
        parser.add_argument("--remove_xml_in_json", action="store_true", help="Remove xml markup from text if input is UZH json.")
        parser.add_argument("--remove_linebreaks_in_json", action="store_true", help="Remove linebreaks from text if input is UZH json.")
        parser.add_argument("--prefix", type=str, default=None, help="Task prefix, will prepend the this prefix to all samples (only when fine-tuning on a single task). ")
        parser.add_argument("--remove_special_tokens_containing", type=str, nargs="+", help="Remove tokens from the special_tokens_map that contain this string (e.g. xml tags).")

        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=512, help="maximum num of wordpieces/summary. Used for training and testing")

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
        parser.add_argument("--lr_scheduler", type=str, default="cosine", help="torch scheduler: 'cosine': CosineAnnealingLR, cosine_wr: CosineAnnealingWarmRestarts ,'plateau': ReduceLROnPlateau")
        parser.add_argument("--cosine_Tmax", type=int, default=10, help="t_max (maximum number of iterations/epochs) for cosine annealing scheduler.")
        parser.add_argument("--cosine_T_mult", type=int, default=1, help="t_mult (factor that increases T_i after each restart) for cosine annealing scheduler with warm restarts. Default: 1.")
        parser.add_argument("--cosine_T_0", type=int, default=1000, help="t_0 (number of iterations for the first restart) for cosine annealing scheduler with warm restarts. Default: 1000.")
        parser.add_argument("--lr", type=float, default=0.00003, help="Initial learning rate")
        parser.add_argument("--check_val_every_n_epoch", type=int, default=None, help="How often to check the validation set in number of epochs.")
        parser.add_argument("--val_check_interval", type=int, help="How often to check the validation set in number of updates.")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--train_percent_check", default=1.00, type=float, help='Percent of training data used (for testing) NOTE: not available in pytprch lightning==1.1.6')
        parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs (will stop training even if patience for early stopping has not been reached). Default: 100.")
        parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps (will stop training even if patience for early stopping has not been reached). Default: -1 (no maximum).")
        parser.add_argument("--early_stopping_metric", type=str, default='rougeL', help="Metric to be used for early stopping: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu")
        parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
        parser.add_argument("--lr_reduce_patience", type=int, default=8, help="Patience for LR reduction in Plateau scheduler. NOTE: if interval=steps, and lr_scheduler=ReduceLROnPlateau, frequency MUST be smaller than the number of batches per epoch, otherwise lr_scheduler.step() never gets called and lr is not reduced (because lightning calls step() in this case based on batch index, which is reset after each epoch).")
        parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Learning rate reduce factor for Plateau scheduler.")
        parser.add_argument("--lr_cooldown", type=int, default=0, help="Cooldown for Plateau scheduler (number of epochs to wait before resuming normal operation after lr has been reduced.).")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--save_top_k", type=int, default=5, help="Number of best checkpoints to keep. Others will be removed.")
        parser.add_argument("--save_every_n_val_epochs", type=int, default=0, help="Number of validation epochs between checkpoints.")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')

        ## inference params
        parser.add_argument("--beam_size", type=int, default=4, help="Beam size for inference when testing/validating. Default: 4.")
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')

        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--disable_validation_bar", action='store_true', help="Do not print tqdm bar on validation.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")
        parser.add_argument("--wandb", type=str, default=None, help="WandB project name to use if logging fine-tuning with WandB.")
        parser.add_argument("--wandb_entity", type=str, default=None, help="WandB account name to use if logging fine-tuning with WandB.")

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = T5Trainer(args)

    if args.print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)


    train_set = T5Dataset(src_file=args.train_source,
                          tgt_file=args.train_target,
                          name="train",
                          tokenizer=model.tokenizer,
                          max_input_len=args.max_input_len,
                          max_output_len=args.max_output_len,
                          prefix=args.prefix)

    dev_set = T5Dataset(src_file=args.dev_source,
                        tgt_file=args.dev_target,
                        name="dev",
                        tokenizer=model.tokenizer,
                        max_input_len=args.max_input_len,
                        max_output_len=args.max_output_len,
                        prefix=args.prefix)


    test_set=None
    if args.test_source is not None:
        test_set = T5Dataset(src_file=args.test_source,
                            tgt_file=args.test_target,
                            name="test",
                            tokenizer=model.tokenizer,
                            max_input_len=args.max_input_len,
                            max_output_len=args.max_output_len,
                            prefix=args.prefix)

    model.set_datasets(train_set=train_set, dev_set=dev_set, test_set=test_set)

    # print validation source and reference to model directory
    validation_source_file=os.path.join(args.save_dir, args.save_prefix, "validation_source")
    validation_reference_file=os.path.join(args.save_dir, args.save_prefix, "validation_reference")
    os.makedirs(os.path.dirname(validation_source_file), exist_ok=True)

    with open(validation_source_file, 'w') as f:
        for line in model.dev_set.inputs:
            if args.dev_jsons is not None:
                line = line[1].replace('\n', ' ')
            f.write(line.strip() + "\n")
    with open(validation_reference_file, 'w') as f:
        for line in model.dev_set.labels:
            if args.dev_jsons is not None:
                line = line[1].replace('\n', ' ')
            f.write(line.strip() + "\n")

    # if test set was set, print source and reference for test as well
    if args.test_source is not None or args.test_jsons is not None:
        test_source_file=os.path.join(args.save_dir, args.save_prefix, "test_source")
        os.makedirs(os.path.dirname(test_source_file), exist_ok=True)
        with open(test_source_file, 'w') as f:
            for line in model.test_set.inputs:
                if args.test_jsons is not None:
                    line = line[1].replace('\n', ' ')
                f.write(line.strip() + "\n")

    if args.test_target is not None or args.test_jsons is not None:
        test_reference_file=os.path.join(args.save_dir, args.save_prefix, "test_reference")
        os.makedirs(os.path.dirname(test_source_file), exist_ok=True)
        with open(test_reference_file, 'w') as f:
            for line in model.test_set.labels:
                if args.test_jsons is not None:
                    line = line[1].replace('\n', ' ')
                f.write(line.strip() + "\n")

    if args.wandb:
        logger = WandbLogger(project=args.wandb, entity=args.wandb_entity)
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(args.save_dir, args.save_prefix), name="tensorboard_logs")

    print(args)

    model.lr_mode='max'
    if args.early_stopping_metric == 'vloss':
        model.lr_mode='min'
    early_stop_callback = EarlyStopping(monitor=args.early_stopping_metric, min_delta=args.min_delta, patience=args.patience, verbose=True, mode=model.lr_mode) # metrics: val_loss, bleu, rougeL

    if args.check_val_every_n_epoch:
        checkpoint_name = "{{epoch:02d}}_{{{}".format(args.early_stopping_metric)
        checkpoint_name += ':.3f}'
    else:
        checkpoint_name = "{{epoch:02d}}_{{step:02d}}_{{{}".format(args.early_stopping_metric)
        checkpoint_name += ':.3f}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix),
        filename=checkpoint_name,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.early_stopping_metric,
        mode=model.lr_mode)

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    if args.disable_validation_bar:
        progress_bar_callback = LitProgressBar(refresh_rate=args.progress_bar_refresh_rate)
    else:
        progress_bar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None,
                         max_epochs=args.max_epochs,
                         max_steps=args.max_steps,
                         accumulate_grad_batches=args.grad_accum,
                         num_sanity_val_steps=args.num_sanity_val_steps,
                         val_check_interval=args.val_check_interval,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         limit_val_batches=args.val_percent_check,
                         limit_test_batches=args.test_percent_check,
                         logger=logger,
                         precision=32 if args.fp32 else "16-mixed",
                         enable_progress_bar=True,
                         callbacks=[early_stop_callback, checkpoint_callback, progress_bar_callback, lr_monitor_callback]
                         )
    ## write config + tokenizer to save_dir
    model.model.save_pretrained(args.save_dir + "/" + args.save_prefix)

    if args.remove_special_tokens_containing:
        print("special tokens before:", model.tokenizer.special_tokens_map)
        model.tokenizer = remove_special_tokens(model.tokenizer, args.remove_special_tokens_containing)
        print("special tokens after:", model.tokenizer.special_tokens_map)

    model.tokenizer.save_pretrained(args.save_dir + "/" + args.save_prefix)
    if args.resume_ckpt: # load complete checkpoint, weights, optimizer + lr_scheduler states
        trainer.fit(model, ckpt_path=args.resume_ckpt)
    elif args.pretrained_ckpt: # load parameter weights, but not optimizer/lr_scheduler states
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        trainer.fit(model)
    else:
        trainer.fit(model)

    if test_set is not None:
        trainer.test(model)
    print("Training ended. Best checkpoint {}.".format(trainer.checkpoint_callback.best_model_path))

    if args.lora:
        save_path=f"{args.save_dir}/{args.save_prefix}/lora_merged"
        print(f"saving merged model to.. {save_path}")
        merged_model = model.model.merge_and_unload()
        os.mkdir(save_path)
        merged_model.save_pretrained(save_path)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="Fine-tune T5 type models")
    parser = T5Trainer.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)

