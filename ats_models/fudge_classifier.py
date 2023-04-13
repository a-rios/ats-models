#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is based on SimpleFUDGE: https://github.com/ZurichNLP/SimpleFUDGE

Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

"""

import os
import argparse
import random
import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from transformers import AutoTokenizer
from torchmetrics import  Accuracy
from typing import Optional
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from functools import partial

from .data import FudgeDatasetJson
from .finetune_mbart import LitProgressBar

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FudgeLSTM(pl.LightningModule):
    def __init__(self, params, tokenizer):
        super().__init__()
        self.args = params
        self.tokenizer = tokenizer

    def init_model(self):
        cp = torch.load(os.path.join(self.args.pretrained_model_path, "pytorch_model.bin"))
        self.embed = torch.nn.Embedding.from_pretrained(cp['model.encoder.embed_tokens.weight'], padding_idx=1)
        del cp
        if self.args.rnn_bidirectional:
            self.rnn = torch.nn.LSTM(1024, self.args.num_hidden//2, num_layers=self.args.num_layers, bidirectional=True, dropout=self.args.dropout)
        else:
            self.rnn = torch.nn.LSTM(1024, self.args.num_hidden, num_layers=self.args.num_layers, bidirectional=False, dropout=self.args.dropout)
        self.out_linear = torch.nn.Linear(self.args.num_hidden, 1)

    def forward(self, inputs):
        lengths = torch.where(inputs == self.tokenizer.pad_token_id, 0, 1).sum(dim=1)
        inputs = self.embed(inputs) # batch x seq x hidden_dim
        inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
        rnn_output, _ = self.rnn(inputs)
        rnn_output, _ = pad_packed_sequence(rnn_output)
        rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x hidden_dim
        return self.out_linear(rnn_output).squeeze(2), lengths # batch x seq


class FudgeClassifier(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_path)

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        self.current_checkpoint =0
        self.best_checkpoint = None
        self.best_metric = 10000 if self.args.monitored_metric == 'vloss' else 0 ## keep track of best dev value of whatever metric is used in early stopping callback
        self.num_not_improved = 0
        self.save_hyperparameters()
        self.init_model()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.set_metrics()
        self.validation_step_outputs = []

    def init_model(self):
        self.model = FudgeLSTM(self.args, self.tokenizer)
        self.model.init_model()

    def set_datasets(self, train_set: FudgeDatasetJson, dev_set: FudgeDatasetJson, test_set: Optional[FudgeDatasetJson] = None):
        self.train_set = train_set
        self.dev_set = dev_set
        if test_set is not None:
            self.test_set = test_set

    def set_metrics(self):
        self.accuracy =  Accuracy(task="binary")

    def create_padding_mask(self, lengths: torch.LongTensor):
        """
        Code from https://github.com/ZurichNLP/SimpleFUDGE/blob/e8bed23b13a2a108dff75beb8ba32e8684fe42f2/util.py#L53
        Create a mask of seq x batch where seq = max(lengths), with 0 in padding locations and 1 otherwise.
        """
        # lengths: bs. Ex: [2, 3, 1]
        max_seqlen = torch.max(lengths)
        expanded_lengths = lengths.unsqueeze(0).repeat((max_seqlen, 1))  # [[2, 3, 1], [2, 3, 1], [2, 3, 1]]
        indices = torch.arange(max_seqlen).unsqueeze(1).repeat((1, lengths.size(0))).to(lengths.device)  # [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        return expanded_lengths > indices  # pad locations are 0. #[[1, 1, 1], [1, 1, 0], [0, 1, 0]]. seqlen x bs # lengths: bs. Ex: [2, 3, 1]

    def forward(self, inputs, labels):
        predictions, lengths = self.model(inputs) # predictions: (batch_size, seq_len)
        if labels is not None:
            labels = labels.unsqueeze(1).expand(-1, predictions.shape[1]).to(dtype=labels.dtype) # (batch_size, seq_len): learn for all positions at once
            padding_mask = self.create_padding_mask(lengths).permute(1,0)
            loss = self.bce(predictions.flatten()[padding_mask.flatten()==1], labels.flatten().float()[padding_mask.flatten()==1])
            return loss, predictions
        else: # during inference
            return predictions

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output[0]

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        self.log('train-loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        inputs, labels = batch
        accuracy = self.accuracy(outputs[1], labels.unsqueeze(1).expand(-1, outputs[1].shape[1]).to(dtype=outputs[1].dtype))

        self.log('vloss', vloss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        scores = {'vloss' : vloss,
                  'accuracy' : accuracy
                  }

        self.validation_step_outputs.append(scores)
        return scores

    def on_validation_epoch_end(self):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'accuracy']
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

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        interval = "step" if args.val_check_interval is not None else "epoch"
        frequency = args.val_check_interval if args.val_check_interval is not None else args.check_val_every_n_epoch

        if self.args.lr_scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode=self.lr_mode,
                factor=self.args.lr_reduce_factor,
                patience=self.args.lr_reduce_patience,
                cooldown=self.args.lr_cooldown,
                verbose=True)

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
                "monitor": self.args.monitored_metric,
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
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
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
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "CosineAnnealingLR",
            }
        else:
            print(f"Unsupported learning rate scheduler: {self.args.lr_scheduler}. Use one of 'cosine', 'plateau'.")
            exit(1)

        return [self.optimizer], [lr_scheduler_config]

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
                          collate_fn=partial(FudgeDatasetJson.collate_fn, pad_token_id=self.tokenizer.pad_token_id))

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'dev', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

# TODO maybe better use another scheduler e.g. self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the generator model directory (need to load tokenizer/initialize embeddings).")
        parser.add_argument("--train_jsons", type=str, nargs='+', required=True, default=None,  help="Path to UZH json file(s) with training data.")
        parser.add_argument("--dev_jsons", type=str, nargs='+', required=True, default=None,  help="Path to UZH json file(s) with dev data.")
        parser.add_argument("--test_jsons", type=str, nargs='+', default=None,  help="Path to UZH json file(s) with test data.")
        parser.add_argument("--print_train", action='store_true',  help="Print training data to train.pos/neg in model folder for debugging.")
        parser.add_argument("--remove_xml_in_json", action="store_true", help="Remove xml markup from text if input is UZH json.")
        parser.add_argument("--remove_linebreaks_in_json", action="store_true", help="Remove linebreaks from text if input is UZH json.")
        parser.add_argument("--tgt_lang", type=str, required=True, default=None, help="If multiple target languages in json files, use tgt_lang as the True class.")
        parser.add_argument("--max_input_len", type=int, default=4096, help="maximum number of wordpieces in input. Longer sequences will be truncated. Default: 4096.")
        parser.add_argument("--min_input_len", type=int, default=3, help="minimum number of words in input (full words, not pieces). Shorter sequences will be discarded. Default: 3.")


        parser.add_argument("--save_dir", type=str, metavar='PATH', required=True, help="Directory to save model.")
        parser.add_argument("--save_prefix", type=str, default='fudge', help="subfolder in save_dir for this model")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")

        parser.add_argument("--num_sanity_val_steps", type=int, default=0,  help="Number of evaluation sanity steps to run before starting the training. Default: 0.")
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulation steps.")
        parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator for pytorch lightning trainer (gpu or cpu).")
        parser.add_argument("--devices", type=int, nargs="+", required=True, help="Device id(s).")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum change in the monitored quantity to qualify as an improvement.")
        parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
        parser.add_argument("--check_val_every_n_epoch", type=int, default=None, help="Number of training steps between validations in epochs.")
        parser.add_argument("--val_check_interval", type=int, help="How often to check the validation set in number of updates.")
        parser.add_argument("--val_percent_check", type=int, help='Percent of validation data used')
        parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs (will stop training even if patience for early stopping has not been reached). Default: 100.")
        parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps (will stop training even if patience for early stopping has not been reached). Default: -1 (no maximum).")
        parser.add_argument("--monitored_metric", type=str, default='accuracy', help="Metric to be used for early stopping: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu")
        parser.add_argument("--patience", type=int, default=None, help="Patience for early stopping.")
        parser.add_argument("--lr_scheduler", type=str, default='cosine', help="Learning Rate scheduler (either 'plateau', 'cosine' (=ReduceLROnPlateau or CosineAnnealingLR).")
        parser.add_argument("--lr_reduce_patience", type=int, default=8, help="Patience for LR reduction in Plateau scheduler.")
        parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Learning rate reduce factor for Plateau scheduler.")
        parser.add_argument("--lr_cooldown", type=int, default=0, help="Cooldown for Plateau scheduler (number of epochs to wait before resuming normal operation after lr has been reduced.).")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--save_top_k", type=int, default=1, help="Number of best checkpoints to keep. Others will be removed.")
        parser.add_argument("--save_every_n_val_epochs", type=int, default=0, help="Number of validation epochs between checkpoints.")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--disable_validation_bar", action='store_true', help="Do not print tqdm bar on validation.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")
        parser.add_argument("--wandb", type=str, default=None, help="WandB project name to use if logging fine-tuning with WandB.")
        parser.add_argument("--wandb_entity", type=str, default=None, help="WandB account name to use if logging fine-tuning with WandB.")

        # model args
        parser.add_argument("--rnn_bidirectional", action='store_true', help="Use bidirectional rnn to encode sentences (default: causal).")
        parser.add_argument("--num_hidden", type=int, default=512, help="Rnn hidden dimension.")
        parser.add_argument("--num_layers", type=int, default=3, help="Rnn layers.")
        parser.add_argument("--dropout", type=float, default=0.1, help="Rnn dropout.")


        return parser

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = FudgeClassifier(args)

    if args.print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)

    train_set = FudgeDatasetJson(json_files=args.train_jsons,
                                name="train",
                                tokenizer=model.tokenizer,
                                max_input_len=args.max_input_len,
                                tgt_tag=args.tgt_lang,
                                remove_xml=args.remove_xml_in_json,
                                remove_linebreaks=args.remove_linebreaks_in_json,
                                seed=args.seed)
    if args.print_train:
        os.makedirs(os.path.dirname(os.path.join(args.save_dir, args.save_prefix, "train.pos")), exist_ok=True)
        with open(os.path.join(args.save_dir, args.save_prefix, "train.pos"), 'w') as f:
            for sample in train_set.positive_samples_list:
                f.write(sample[0])
                if args.remove_linebreaks_in_json:
                    f.write("\n")

        with open(os.path.join(args.save_dir, args.save_prefix, "train.neg"), 'w') as f:
            for sample in train_set.negative_samples_list:
                f.write(sample[0])
                if args.remove_linebreaks_in_json:
                    f.write("\n")

    dev_set = FudgeDatasetJson(json_files=args.dev_jsons,
                                name="dev",
                                tokenizer=model.tokenizer,
                                max_input_len=args.max_input_len,
                                tgt_tag=args.tgt_lang,
                                remove_xml=args.remove_xml_in_json,
                                remove_linebreaks=args.remove_linebreaks_in_json,
                                seed=args.seed)
    test_set = None
    if args.test_jsons is not None:
        test_set = FudgeDatasetJson(json_files=args.test_jsons,
                                    name="test",
                                    tokenizer=model.tokenizer,
                                    max_input_len=args.max_input_len,
                                    tgt_tag=args.tgt_lang,
                                    remove_xml=args.remove_xml_in_json,
                                    remove_linebreaks=args.remove_linebreaks_in_json,
                                    seed=args.seed)
    model.set_datasets(train_set=train_set, dev_set=dev_set, test_set=test_set)

    if args.wandb:
        logger = WandbLogger(project=args.wandb, entity=args.wandb_entity)
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(args.save_dir, args.save_prefix), name="tensorboard_logs")

    print(args)
    with open(os.path.join(args.save_dir, args.save_prefix, "arguments.txt"), 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f"{k}: {v}\n")

    model.lr_mode='max'
    if args.monitored_metric == 'vloss':
        model.lr_mode='min'
    early_stop_callback = EarlyStopping(monitor=args.monitored_metric, min_delta=args.min_delta, patience=args.patience, verbose=True, mode=model.lr_mode) # metrics: val_loss, bleu, rougeL

    checkpoint_name = "{{epoch:02d}}_{{{}".format(args.monitored_metric)
    checkpoint_name += ':.3f}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix),
        filename=checkpoint_name,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.monitored_metric,
        mode=model.lr_mode)

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    if args.disable_validation_bar:
        progress_bar_callback = LitProgressBar(refresh_rate=args.progress_bar_refresh_rate)
    else:
        progress_bar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None,
                         max_epochs=args.max_epochs,
                         max_steps=args.max_steps,
                         accumulate_grad_batches=args.grad_accum,
                         num_sanity_val_steps=args.num_sanity_val_steps,
                         val_check_interval=args.val_check_interval,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         limit_val_batches=args.val_percent_check,
                         limit_test_batches=1.0,
                         logger=logger,
                         precision=32 if args.fp32 else "16-mixed",
                         enable_progress_bar=True,
                         callbacks=[early_stop_callback, checkpoint_callback, progress_bar_callback, lr_monitor_callback]
                         )
    if args.resume_ckpt: # load complete checkpoint, weights, optimizer + lr_scheduler states
        trainer.fit(model, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(model)

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="Train Fudge classifiers")
    parser = FudgeClassifier.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
