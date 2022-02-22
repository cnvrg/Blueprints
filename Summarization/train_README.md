# Blueprint : Training (Generating the summary of articles after fine tuning a pretrained model)
This blueprint has two libraries
1. **Text Summarization Train**
2. **Text Summarization Inference**

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
## Text Summarization Train
This library serves as a tool for fine-tuning a pretrained summarization model (for getting abstractive summarizations) on a pre-built dataset or a custom dataset given by the user. The dataset, wiki_lingua is actually referenced from huggingface library and contains upto 3,500 rows of articles and summaries, in English language. It’s up to the user to specify the number of rows on which they want to train their model. 
## Text Summarization Inference
This library serves as a tool for getting abstractive summarization of English articles without training the model further. It uses a specific model trained by CNVRG on custom data (wiki_lingua dataset) and gives summaries of around 7% of the total article size. While running this library, the user needs to give the following parameters: -
## What can you expect?
- abstractive summaryof any article


## What you need to provide?
- training file
- input fie
- hyper paramters


### Model Used
is a fine-tuned version of [bart_large_cnn](https://huggingface.co/facebook/bart-large-cnn) from [AutoModelSeq2Seq](https://huggingface.co/transformers/model_doc/encoderdecoder.html) class of [transformers](https://huggingface.co/transformers/) library. The function used is model.generate() and the summary length is restricted to 500 words as well and is always higher than 7% of the article length.
BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.
BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.
### Trainer
The Trainer class provides an API for feature-complete training in PyTorch for most standard use cases. It’s used in most of the example scripts.
Before instantiating your Trainer, create a TrainingArguments to access all the points of customization during training.
The API supports distributed training on multiple GPUs/TPUs, mixed precision through NVIDIA Apex and Native AMP for PyTorch.
The Trainer contains the basic training loop which supports the above features. To inject custom behavior you can subclass them and override the following methods: