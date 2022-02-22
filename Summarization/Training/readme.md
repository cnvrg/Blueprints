# Fine tuning a huggingface model for predicting summaries
## training_summarization

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This library serves as a tool for fine-tuning a pretrained summarization model (for getting abstractive summarizations) on a pre-built dataset or a custom dataset given by the user. The dataset, wiki_lingua is actually referenced from huggingface library and contains upto 3,500 rows of articles and summaries, in English language. It’s up to the user to specify the number of rows on which they want to train their model. The dataset that the user can upload will look like this: -
This library uses the following arguments: -

- `--training_file` refers to the name of the file (and its path) which contains user’s data to train the model on or the custom wiki_lingua dataset. It would look like this: - training_data/wiki_lingua_file.csv, where training_data is the name of the dataset
   - This file would have 2 columns, one will contain the article (titled document) while the other will contain summary.
   - The headers of the file should be a pair of strings called **document** and **summary**
   - The format should be **CSV** (not text or any other)
   - The dataset which contains the file should be of the text format.
   -    |document   |summary
        |---|---
        |make sure that the area is a safe place......   |walk to the area....   
        |Designating a driver is a very popular..|driver is hired from..
 - `--number_of_rows` refers to the number of rows of the wiki_lingua dataset on which the model will be trained or the number of rows in the custom user dataset that the user wants to use for the training. (Recommended <500 rows for quicker results)
 - `--Trained Model Name` refers to the name which the user wants to give to their custom model
 - `--default_model` refers to the model 'bart large cnn' on which the training is performed
 - `--encoder_max_length` refers to the maximum length into which the data is encoded
 - `--decoder_max_length` refers to the maximum length of the sentences into which the model decodes the encoded sequences
 - `--label_smooth_factor` refers to the factor which smoothes the labels. Zero means no label smoothing, otherwise the underlying onehot-encoded labels are changed from 0s and 1s to label_smoothing_factor/num_labels and 1 - label_smoothing_factor + label_smoothing_factor/num_labels respectively.
 - `--weight_decay_factor` refers to the fac
 
## How to run

```
cnvrg run  --datasets='[{id:"summarization_train_dataset",commit:"3fcfb99ec010d4a8ba364f43169465d91ca39ada"}]' --machine="default.Large,default.large - spot" --image=cnvrg:v5.0 --sync_before=false python3 training_model_summary.py --Trained_Model_Name my_custom_model --training_file summarization_train_dataset/wiki_lingua_file.csv --train_rows 80 --default_model /cnvrg/Model/bart_large_cnn_original_1/ --encoder_max_length 256 --decoder_max_length 64 --label_smooth_factor 0.1 --weight_decay_factor 0.1
```

### Model Used
is a fine-tuned version of [bart_large_cnn](https://huggingface.co/facebook/bart-large-cnn) from [AutoModelSeq2Seq](https://huggingface.co/transformers/model_doc/encoderdecoder.html) class of [transformers](https://huggingface.co/transformers/) library. The function used is model.generate() and the summary length is restricted to 500 words as well and is always higher than 7% of the article length.
BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.
BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.
### Trainer
The Trainer class provides an API for feature-complete training in PyTorch for most standard use cases. It’s used in most of the example scripts.
Before instantiating your Trainer, create a TrainingArguments to access all the points of customization during training.
The API supports distributed training on multiple GPUs/TPUs, mixed precision through NVIDIA Apex and Native AMP for PyTorch.
The Trainer contains the basic training loop which supports the above features. To inject custom behavior you can subclass them and override the following methods: