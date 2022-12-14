#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

prepare_long_input() adapted from:
    https://github.com/allenai/longformer/
"""

import torch
from torch.utils.data import DataLoader, Dataset
import json, os
import argparse
from typing import Optional, List
from bs4 import BeautifulSoup
from .data import CustomDatasetUZHJson


def main(args):

    data_set = CustomDatasetUZHJson(json_files=[args.json],
                              name="data",
                              tokenizer=None,
                              max_input_len=4096,
                              max_output_len=1024,
                              src_lang=args.src_lang,
                              tgt_lang=args.tgt_lang,
                              remove_xml=args.remove_xml_in_json
                )

    source_file= args.out_file + "." + args.src_lang
    reference_file= args.out_file + "." + args.tgt_lang
    os.makedirs(os.path.dirname(source_file), exist_ok=True)

    with open(source_file, 'w') as f:
        for line in data_set.inputs:
            line = line[1].replace('\n', ' ')
            line = line.replace('  ', ' ')
            f.write(line + "\n")
    with open(reference_file, 'w') as f:
        for line in data_set.labels:
            line = line[1].replace('\n', ' ')
            line = line.replace('  ', ' ')
            f.write(line + "\n")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help="Path to uzh json file.")
    parser.add_argument('--out_file', type=str, required=True, help="Path to output text files.")
    parser.add_argument("--src_lang", type=str, default=None, help="Source language tag (optional, for multilingual batches, preprocess text files to include language tags.")
    parser.add_argument("--tgt_lang", type=str, default=None, help="Target language tag (optional, for multilingual batches, preprocess text files to include language tags.")
    parser.add_argument("--remove_xml_in_json", action="store_true", help="Remove xml markup from text if input is UZH json.")
    args = parser.parse_args()

    main(args)
