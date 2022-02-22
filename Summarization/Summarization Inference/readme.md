# Generating the summary of a single articles via a fine-tuned huggingface model
## Summarization Inference

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This library serves as a tool for getting abstractive summarization of a single English article at a time, without training the model further. It uses a specific model trained by CNVRG on custom data (wiki_lingua dataset) and gives summaries of around 7% of the total article size. While running this library, the user needs to give the following parameter: -
## Arguments
- `--data` refers to the paragraph, which needs to be summarized. Be careful to not introduce any apostrophes or other punctuation marks in the paragraph.

### Output
- Summary Text

### Model Used
is a fine-tuned version of [bart_large_cnn](https://huggingface.co/facebook/bart-large-cnn) from [AutoModelSeq2Seq](https://huggingface.co/transformers/model_doc/encoderdecoder.html) class of [transformers](https://huggingface.co/transformers/) library. The function used is model.generate() and the summary length is restricted to 500 words as well and is always higher than 7% of the article length.
BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.
BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.