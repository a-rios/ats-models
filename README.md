# Models for Automatic Text Simplification (ATS)
This repository contains code for the ATS models we use in the [Flagship Inclusive Information and Communication Technologies (IICT) project](https://www.iict.uzh.ch/en.html).

### Vocabulary lists for German
The folder [vocab_lists](vocab_lists) contains lists of the N most frequent German subwords in the mBART vocabulary. These lists were created by tokenizing ~2 million German sentences from the following corpora:
* [Europarl-v10](https://www.statmt.org/europarl/v10/training-monolingual/europarl-v10.de.tsv.gz)
* [News Commentary v16](https://data.statmt.org/news-commentary/v16/training-monolingual/news-commentary-v16.de.gz)
* the first 2 million sentences from [Common Crawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/de.xz)

These vocabulary lists can be used with the provided conversion scripts to reduce mBART's orginial vocabulary of ~250k tokens to mostly German subwords, see below.

### Trimming and Conversion scripts


### Fine-Tuning

### Inference

