# Models for Automatic Text Simplification (ATS)
This repository contains code for the ATS models we use in the [Flagship Inclusive Information and Communication Technologies (IICT) project](https://www.iict.uzh.ch/en.html).
This repository is an updated version of [longmbart](https://github.com/a-rios/longmbart), which is in turn based on [longformer](https://github.com/allenai/longformer) and [huggingface transformers](https://github.com/huggingface/transformers).
The code in this repository includes scripts to trim and fine-tune models based on mt5 [[2]](#[2]) and mBART [[3]](#[3]), and in case of the latter, an optional modification to use longformer attention [[1]](#[1]) in the encoder for long sequences. 

Content:
- [Installation](#installation)
  * [Vocabulary lists for German](#vocabulary-lists-for-german)
  * [Trimming and Conversion scripts](#trimming-and-conversion-scripts)
    + [mt5](#mt5)
    + [mBART](#mbart)
    + [Longmbart](#longmbart)
  * [Fine-Tuning](#fine-tuning)
  * [mBART with standard attention example:](#mbart-with-standard-attention-example)
  * [mBART with longformer attention example:](#mbart-with-longformer-attention-example)
  * [Inference](#inference)
  * [Citation](#citation)
- [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## Installation
Check out the repository:
```
git clone https://github.com/a-rios/ats-models.git
cd ats-models
pip install -e .
```

### Vocabulary lists for German
The folder [vocab_lists](vocab_lists) contains lists of the N most frequent German subwords in the mBART and mt5 vocabulary. These lists were created by tokenizing ~2 million German sentences from the following corpora:
* [Europarl-v10](https://www.statmt.org/europarl/v10/training-monolingual/europarl-v10.de.tsv.gz)
* [News Commentary v16](https://data.statmt.org/news-commentary/v16/training-monolingual/news-commentary-v16.de.gz)
* the first 2 million sentences from [Common Crawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/de.xz)

These vocabulary lists can be used with the provided conversion scripts to reduce mBART/mt5's orginial vocabulary of >200k tokens to mostly German subwords, see below.

### Trimming and Conversion scripts
The scripts [trim_mbart.py](ats_models/trim_mbart.py) and [trim_mt5.py](ats_models/trim_mt5.py) can be used to trim the vocabularies of the respective pretrained model. 

#### mt5
```
python -m ats_models.trim_mt5 \
--base_model "google/mt5-small" \
--tokenizer "google/mt5-small" \
--save_model_to path-to-save-the-trimmed-model \
--cache_dir path-to-huggingface-cache-dir \
--reduce_to_vocab path-to-your-vocab-list
```
If you want to use the larger pretrained models, just replace `mt5-small` with `mt5-base` or `mt5-large` in `--base_model/tokenizer`.

#### mBART
```
python -m ats_models.trim_mbart \
--base_model facebook/mbart-large-cc25" \
--tokenizer facebook/mbart-large-cc25" \
--save_model_to path-to-save-the-trimmed-model \
--cache_dir path-to-huggingface-cache-dir \
--reduce_to_vocab path-to-your-vocab-list
--add_language_tags de_A1 de_A2 de_B1 \
--initialize_tags de_DE de_DE de_DE 
```
`--cache_dir` is optional, per default, huggingface will download and save models in the user's home directory, see https://huggingface.co/docs/datasets/cache.

For German, you can use the vocabulary lists provided in [vocab_lists](vocab_lists), for other languages, you have to create your own (one subword per line).

The script for mBART has some extra options to add new language tags, e.g. CEFR levels for text simplification. The embedding of these new tags can be initialized with an embedding of one of the pretrained tags, e.g. `de_DE`(if not set, they will be randomly initialized).
The mBART conversion script also offers an option to add a list of additional items to the vocabulary (`--add_to_vocab`). 

#### Longmbart
Longmbart is mBART but with longformer attention in the encoder [1]:

```
python -m ats_models.convert_mbart2long \
--base_model facebook/mbart-large-cc25" \
--tokenizer facebook/mbart-large-cc25" \
--save_model_to path-to-save-the-trimmed-model \
--cache_dir path-to-huggingface-cache-dir \
--max_pos max-source-positions (default: 4096)\
--attention_window attention-window-size (default: 512)\
--reduce_to_vocab path-to-your-vocab-list
--add_language_tags de_A1 de_A2 de_B1 \
--initialize_tags de_DE de_DE de_DE 
```

### Fine-Tuning
The repository currently only has scripts for fine-tuning mBART based models, mt5 fine-tuning might be added in the future. Some general information on options:

* language tags: mBART requieres source and target language tags:
   * source and target language are fixed, i.e. all source and all target samples are in the same language : 
     ```
     --src_lang en_XX \
     --tgt_lang de_DE \
     ```
    * tags are included in the text files as the first token in each sample:
  
      | source        | target      |
      | ------------- |-------------|
      | en_XX This is an example. | de_DE Dies ist ein Beispiel. |
     
       ```
       --src_tags_included \
       --tgt_tags_included \
       ```
    * options can be mixed, you can have a single source language and mixed target samples, in this case use:
       ```
       --src_lang your-lang \
       --tgt_tags_included \
       ```
    * if source or target is one language, setting `--src_lang` or `--tgt_lang` will be slightly faster than reading the tags from the text
    * input can also be given as json files, however, currently the only supported format is the internal UZH json. The option `--remove_xml_in_json` will remove the xml markup to train a model on text only, without this option, the model will be trained to predict text with xml layout information according to the content of the json files. If using json as input, there is no need to specify source and target languages, as those are read from the json file itself.
       ```
       --train_jsons list-of-json-files \
       --dev_jsons list-of-json-files \
       --test_jsons list-of-json-files \
       --remove_xml_in_json (optional) \
       ```
* `--num_workers`: affects dataloader, depends on dataset size and available CPU, see [Pytorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html)
* the script has an option `--fp32` to fine-tune in fp32, default is fp16.
* metrics supported for early stopping are: `vloss, rouge1, rouge2, rougeL, rougeLsum, bleu`
* if you added special tokens to the vocabulary, you can remove them for the validation evaluation with `--remove_special_tokens_containing`. This is useful e.g. for xml tags if you do early stopping on rouge* or bleu scores.


### mBART with standard attention example:

```
python -m ats_models.finetune_mbart \
--from_pretrained path-to-trimmed-mbart \
--tokenizer path-to-trimmed-mbart \
--save_dir path-to-save-the-finetuned-model \
--save_prefix name-of-model \
--train_source train-file.src \
--train_target train-file.trg \
--dev_source dev-file.src \
--dev_target dev-file.trg \
--test_source test-file.src \
--test_target test-file.trg \
--max_input_len max-input-len (cannot be longer than 1024) \
--max_output_len max-output-len (cannot be longer than 1024)  \
--src_lang "de_DE" \
--tgt_tags_included \
--batch_size batch-size \
--grad_accum number-of-updates-for-gradient-accumulation \
--num_workers number-of-workers \
--accelerator gpu \
--devices device-id \
--seed seed-number \
--attention_dropout 0.1 \
--dropout 0.3 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'vloss' \
--patience 10 \
--min_delta 0.0005 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10 \
--disable_validation_bar \
--save_top_k 2

combination of 2 of these options:
--src_lang src-lang \
--tgt_lang tgt-lang \
--tgt_tags_included \
--tgt_tags_included \
```
### mBART with longformer attention example:
mBART with longformer attention has some additional options:
* `--attention_mode` should be set to `sliding_chunks` (default is `n2`, which is the standard mBART attention)
* `--attention_window`: the window size for full attention, see [1] for details
* `--global_attention_indices`: optional, the indices that always have full attention. Default is to have full attention on the last non-padded position of the source (=language tag).

```
python -m ats_models.finetune_mbart \
--from_pretrained path-to-trimmed-mbart \
--tokenizer path-to-trimmed-mbart \
--save_dir path-to-save-the-finetuned-model \
--save_prefix name-of-model \
--train_source train-file.src \
--train_target train-file.trg \
--dev_source dev-file.src \
--dev_target dev-file.trg \
--test_source test-file.src \
--test_target test-file.trg \
--max_output_len max-output-len (cannot be longer than 1024) \
--max_input_len max-input-len (cannot be longer than what was set in the conversion script) \
--attention_mode sliding_chunks \
--attention_window window-size-for-full-attention (default: 512) \
--batch_size batch-size \
--grad_accum number-of-updates-for-gradient-accumulation \
--num_workers number-of-workers \
--accelerator gpu \
--devices device-id \
--seed seed-number \
--attention_dropout 0.1 \
--dropout 0.3 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'vloss' \
--patience 10 \
--min_delta 0.0005 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--disable_validation_bar \
--progress_bar_refresh_rate 10

combination of 2 of these options:
--src_lang src-lang \
--tgt_lang tgt-lang \
--tgt_tags_included \
--tgt_tags_included \
```

### Inference

Translating with a fine-tuned model is done with `inference_mbart.py`. If a reference translation is given, the script will calculate automatic metrics with [rouge_score](https://github.com/google-research/google-research/tree/master/rouge) and [sacrebleu](https://github.com/mjpost/sacrebleu). 
The model needs a language tag for each sample, this can be set in the following ways:

* source and target language are fixed, i.e. all source and all target samples are in the same language: 
     ```
     --src_lang en_XX \
     --tgt_lang de_DE \
     ```
* tags are included in the text files as the first token in each sample:
  
      | source        | target      |
      | ------------- |-------------|
      | en_XX This is an example. | de_DE Dies ist ein Beispiel. |
     
       ```
       --src_tags_included \
       --tgt_tags_included \
       ```
 * if no reference is available but the target languages are mixed, target tags can be given as a separate input file, one tag per line (has to be same length as source text):
      | source        | target      |
      | ------------- |-------------|
      | en_XX This is an example. | de_DE  |
      | en_XX This is another example. | es_XX |
      
       ```
       --target_tags file-with-tags \
       ```
Options can be mixed, same as for training, e.g. set one source language with `--src_lang` but different target languages with either `--tgt_tags_included` or `--target_tags`.

Example for translating with mbart:
```
python -m ats_models.inference_mbart \
--model_path path-to-fine-tuned-mbart \
--checkpoint checkpoint-name \
--tokenizer path-to-fine-tuned-mbart \
--test_source file-to-be-translated \
--src_lang en_XX \
--tgt_lang de_DE \
--max_input_len max-input-length \
--max_output_len max-output-length \
--batch_size batch-size \
--num_workers 1 \
--accelerator gpu \
--devices device-id \
--beam_size beam-size \
--progress_bar_refresh_rate 1 \
--translation path-to-output-file
```
The arguments for translating with a longmbart model are identical, but it needs the additional switch `--is_long`, otherwise the model will not be loaded correctly.

### Citation
If you use code in this repository, please cite the following publication:

Annette Rios, Nicolas Spring, Tannon Kew, Marek Kostrzewa, Andreas Säuberli, Mathias Müller, and Sarah Ebling. 2021. [A New Dataset and Efficient Baselines for Document-level Text Simplification in German.](https://aclanthology.org/2021.newsum-1.16/) In Proceedings of the Third Workshop on New Frontiers in Summarization, pages 152–161, Online and in Dominican Republic. Association for Computational Linguistics.

Bibtex:
```
@inproceedings{rios-etal-2021-new,
    title = "A New Dataset and Efficient Baselines for Document-level Text Simplification in {G}erman",
    author = {Rios, Annette  and
      Spring, Nicolas  and
      Kew, Tannon  and
      Kostrzewa, Marek  and
      S{\"a}uberli, Andreas  and
      M{\"u}ller, Mathias  and
      Ebling, Sarah},
    booktitle = "Proceedings of the Third Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2021",
    address = "Online and in Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.newsum-1.16",
    doi = "10.18653/v1/2021.newsum-1.16",
    pages = "152--161",
}
```

## References

<div id="[1]">[1] Iz Beltagy and Matthew E. Peters and Arman Cohan (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). CoRR abs/2004.05150.</div>
</br>
<div id="[2]">[2] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: [A Massively Multilingual Pre-trained Text-to-Text Transformer.](https://aclanthology.org/2021.naacl-main.41/) In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics.</div>
</br>
<div id="[3]">[3] Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. [Multilingual Denoising Pre-training for Neural Machine Translation.](https://aclanthology.org/2020.tacl-1.47/) Transactions of the Association for Computational Linguistics, 8:726–742.</div>
