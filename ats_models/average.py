#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code adapted from sockeye.average:
    https://github.com/awslabs/sockeye/blob/main/sockeye/average.py

Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

"""

import os
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import List
import logging
import torch
import re
import copy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def average_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
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
        return sum(tensors) / len(tensors)


def main(model_path: str,
         checkpoints: List[str],
         out_path: str,
         copy_states_from: int):
    all_checkpoints = []  # type: List[Dict[str, torch.Tensor]]
    for path in checkpoints:
        logger.info("Loading parameters from '%s'", path)
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        params = torch.load(os.path.join(model_path,path), map_location=torch.device(map_location))
        all_checkpoints.append(params)

    logger.info("%d models loaded", len(all_checkpoints))
    assert (all(all_checkpoints[0]['state_dict'].keys() == p['state_dict'].keys() for p in all_checkpoints)), "param names do not match across models"

    avg_params = {}
    avg_checkpoint = {}
    # average arg_params
    # pytorch lightning checkpoint keys:
    # epoch
    # global_step
    # pytorch-lightning_version
    # state_dict
    # loops
    # callbacks
    # optimizer_states
    # lr_schedulers
    # NativeMixedPrecisionPlugin
    # hparams_name
    # hyper_parameters

    # copy values copy_states_from to avg_params, then average state_dict params
    copy_checkpoint = copy.deepcopy(all_checkpoints[copy_states_from])
    copy_checkpoint['state_dict'] = None

    for k in all_checkpoints[0]['state_dict']:
        print("", k)
        tensors = [p['state_dict'][k] for p in all_checkpoints]
        avg_params[k] = average_tensors(tensors)

    copy_checkpoint['state_dict'] = avg_params
    torch.save(copy_checkpoint, out_path)
    logger.info("Averaged parameters written to '%s'", out_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, metavar='PATH', required=True, help='Path to model with checkpoints to be averaged.')
    parser.add_argument('--checkpoints', type=str, nargs="+", metavar='PATH', required=True, help='Checkpoint names to be averaged.')
    parser.add_argument('--out_path', type=str, metavar='PATH', required=True, help='Output path to write averaged checkpoint to.')
    parser.add_argument('--copy_states_from', type=int,  required=True, help='Index of checkpoint given with --checkpoints to copy information other than parameters (optimizer_states, epoch, lr_schedulers, etc).')

    args = parser.parse_args()

    main(args.model_path ,args.checkpoints, args.out_path, args.copy_states_from)
