# Models for Automatic Text Simplification (ATS)
This repository contains code for the ATS models we use in the [Flagship Inclusive Information and Communication Technologies (IICT) project](https://www.iict.uzh.ch/en.html).

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

#### longmbart
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
* `--num_workers`: affects dataloader, depends on dataset size and available CPU, see [Pytorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html)


### mBART with standard attention:

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
--devices 0 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'vloss' \
--patience 12 \
--min_delta 0.0005 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10 \
--fp32 \
--disable_validation_bar \
--save_top_k 2
```
### mBART with longformer attention:

### Inference

## References

[1] Iz Beltagy and Matthew E. Peters and Arman Cohan (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). CoRR abs/2004.05150.
