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

The script for mBART has some extra options to add new language tags, e.g. CEFR levels for text simplification. The embedding of these new tags can be initialized an embedding of one of the pretrained tags, e.g. `de_DE`(if not set, they will be randomly initialized).

#### longmbart
Longmbart is mBART but with longformer windowed attention in the encoder:

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

### Inference

