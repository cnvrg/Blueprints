# Generating the summary of articles via a fine-tuned huggingface model
## Predict_Summary

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This library serves as a tool for getting abstractive summarization of English articles without training the model further. It uses a specific model trained by CNVRG on custom data (wiki_lingua dataset) and gives summaries of around 7% of the total article size. While running this library, the user needs to give the following parameters: -

- `--input_path` refers to the name and path of the file which contains the articles. It would look similar to the **output_summaries_file** described in the readme of **wikipedia_connector** library. 
   - This file would have 2 columns, one will contain the summary while the other will contain the dummy value 'x'. That dummy value is purely for formatting purposes and does not need to edited by the user.
   - The headers of the file should be a pair of strings called **document** and **summary** and the format should be **CSV** (not text or any other)
   - The file path would contain the name of the dataset as well which contains the file. It would look like this :- `\dataset_name\input_file.csv`. Ensure that the dataset is of the text format.
   - The values can contain **article text**, cleaned off any special characters.
   - In case the library is being used alongside another library (wikipeda_connectr) then ensure that the input path contains the location of the file that is outputted by the wikipedia connector. Eg: - `--/input/wikipedia_connector/wiki_output_2.csv`
   -    |input   |
        |---|
        |make sure that the area is a safe place......   |   
        |Designating a driver is a very popular..|
- `--default_model` refers to the model that will generate the summaries. By default, it will be the cnvrg model, trained on wiki_lingua dataset. (More detail in the model section). “/Model/Model/” is the default path of the model. In case of a custom model, the path will be changed to “/input/train-task-name/model-file” where “train-task-name” refers to the library that you used to run the code.
- `--output_file_name` refers to the name of the summary file that the user wants to generate after the summarization. This will be one of the model artifacts that the user will be able to download. Below is a snapshot of the output file generated as an artifact
   -    |GeneratedSummary   |
        |---|
        |make sure that the area is a safe place......   |   
        |Designating a driver is a very popular..| 
- `--min_percent` refers to the lowest ratio of the paragraph, to which the summary lengths can be reduced to.
## How to run
```
cnvrg run  --datasets='[{id:"summarization_train_dataset",commit:"50336bc687eb161ee9fb0ddb8cf2b7e65bad865f"}]' --machine="default.medium" --image=tensorflow/tensorflow:latest-gpu --sync_before=false python3 generate_summary.py --output_file_name summary_generated_1.csv --input_path /data/summarization_train_dataset/input.csv --default_model /cnvrg/Model/trained_model_256_64_600/ --min_percent 0.07 --encoder_max_length 256 --decoder_max_length 64
```
### Model Used
is a fine-tuned version of [bart_large_cnn](https://huggingface.co/facebook/bart-large-cnn) from [AutoModelSeq2Seq](https://huggingface.co/transformers/model_doc/encoderdecoder.html) class of [transformers](https://huggingface.co/transformers/) library. The function used is model.generate() and the summary length is restricted to 500 words as well and is always higher than 7% of the article length.
BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.
BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.