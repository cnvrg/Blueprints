# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:59:16 2021

@author: abhay.saini
"""
import argparse
import pandas as pd
import torch
import numpy as np
import datasets
import transformers
from transformers import Trainer, TrainingArguments
import nltk
from pyarrow import csv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument(
    "--output_file_name",
    action="store",
    dest="output_file_name",
    default="summary_generated_1.csv",
    required=False,
    help="""name of the output summaries file""",
)
parser.add_argument(
    "--input_path",
    action="store",
    dest="input_path",
    default="/input/wikipedia_connector/cnvrg/wiki_output_2.csv",
    required=False,
    help="""name of the file containing the wikipedia output""",
)
parser.add_argument(
    "--default_model",
    action="store",
    dest="default_model",
    default="./Model/bart_large_cnn_original_1/",
    required=False,
    help="""cnvrg trained model""",
)
parser.add_argument(
    "--min_percent",
    action="store",
    dest="min_percent",
    default="0.07",
    required=False,
    help="""ratio of minimum length of the summary""",
)
parser.add_argument(
    "--encoder_max_length",
    action="store",
    dest="encoder_max_length",
    default="256",
    required=True,
    help="""hyperparamter while training""",
)
args = parser.parse_args()

language = "english"
address_model_cnvrg = args.default_model
rows_cnt = pd.read_csv(args.input_path).shape[0]
sub1 = "train[:" + str(rows_cnt) + "]"
input_doc = datasets.load_dataset("csv", data_files=args.input_path, split=(str(sub1)))
model_cnvrg = AutoModelForSeq2SeqLM.from_pretrained(address_model_cnvrg)
output_file_name = args.output_file_name
tokenizer = AutoTokenizer.from_pretrained("Tokenizer/")
min_percent = float(args.min_percent)
encoder_max_length = int(args.encoder_max_length)

def generate_summary(test_samples, model):
    outputs_1 = []
    outputs_str_1 = []
    for i in range(len(test_samples)):
        inputs = tokenizer(
            test_samples["document"][i],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
            return_tensors="pt",
        )
        print(i)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        min_length_1 = min_percent * len(test_samples["document"][i])
        #max_length_1 = max_percent * len(test_samples["document"][i])
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=500,
            min_length=round(min_length_1),
        )
        outputs_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs_1.append(outputs)
        outputs_str_1.append(outputs_str)

    return outputs_1, outputs_str_1


print("defined_generate function")


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["document"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )
    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


print("defined tokenize function")
summaries_case_0 = generate_summary(input_doc, model_cnvrg)[1]
print(summaries_case_0)
print("generated summaries")
summaries_generated = pd.DataFrame(summaries_case_0, columns=["Generated_Summary"])
print(summaries_generated)
print("created dataframe")
summaries_generated.to_csv("/cnvrg/{}".format(output_file_name), index=False)
print("outputted summaries")

