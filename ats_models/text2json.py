#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Note:
    Authors: Annette Rios (arios@cl.uzh.ch)

prepare_long_input() adapted from:
    https://github.com/allenai/longformer/
"""

import json, os, re
import argparse


def main(args):

    out = dict()
    segments = []
    # if parallel data: create file pairs
    if args.tgt_suffix is not None:
        filenames = [name.replace(args.src_suffix, '') for name in os.listdir(args.input_dir) if name.endswith(args.src_suffix)]
        for name in filenames:
            src_file = name + args.src_suffix
            tgt_file = name + args.tgt_suffix
            print(src_file)
            with open(os.path.join(args.input_dir, src_file), 'r') as s, open(os.path.join(args.input_dir, tgt_file), 'r') as t:
                src_txt = s.read()
                tgt_txt = t.read()
                src_txt = re.sub('[^\S\n\r]+', ' ', src_txt)
                tgt_txt = re.sub('[^\S\n\r]+', ' ', tgt_txt)
                segment = {args.tgt_tag : re.sub('\n+', '\n', tgt_txt),# replace multiple newlines with one
                           args.src_tag : re.sub('\n+', '\n', src_txt)
                    }
                segments.append(segment)
    else: #monolingual
        for filename in os.listdir(args.input_dir):
            print(filename)
            with open(os.path.join(args.input_dir, filename), 'r') as f:
                text = f.read()
                text = re.sub('[^\S\n\r]+', ' ', text)
                segment = {args.src_tag: re.sub('\n+', '\n', text)}
                segments.append(segment)

    data = {'segments': segments}
    with open(args.json, 'w') as j:
        json.dump(data, j, indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help="Path to output uzh json file.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to directory with input text files.")
    parser.add_argument("--src_tag", type=str, default=None, help="Source language name if parallel data, or language name if monolingual.")
    parser.add_argument("--tgt_tag", type=str, default=None, help="Target language name if parallel data.")
    parser.add_argument("--src_suffix", type=str, default=None, help="Source language file name suffix.")
    parser.add_argument("--tgt_suffix", type=str, default=None, help="Target language file name suffix.")
    args = parser.parse_args()

    main(args)
