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
    "--Trained_Model_Name",
    action="store",
    dest="Trained_Model_Name",
    default="My_Custom_Model",
    required=False,
    help="""name of the trained model """,
)
parser.add_argument(
    "--training_file",
    action="store",
    dest="training_file",
    default="training_data/wiki_lingua_file.csv",
    required=False,
    help="""name of the file containing the data to train the model on""",
)
parser.add_argument(
    "--default_model",
    action="store",
    dest="default_model",
    default="./Model/Model/",
    required=False,
    help="""cnvrg trained model""",
)
parser.add_argument(
    "--train_rows",
    action="store",
    dest="train_rows",
    default="100",
    required=False,
    help="""no of rows on which the model will be trained""",
)
parser.add_argument(
    "--label_smooth_factor",
    action="store",
    dest="label_smooth_factor",
    default="0.1",
    required=False,
    help="""hyperparamter while training""",
)
parser.add_argument(
    "--weight_decay_factor",
    action="store",
    dest="weight_decay_factor",
    default="0.1",
    required=False,
    help="""hyperparamter while training""",
)
parser.add_argument(
    "--encoder_max_length",
    action="store",
    dest="encoder_max_length",
    default="256",
    required=True,
    help="""hyperparamter while training""",
)
parser.add_argument(
    "--decoder_max_length",
    action="store",
    dest="decoder_max_length",
    default="64",
    required=True,
    help="""hyperparamter while training""",
)

args = parser.parse_args()

language = "english"
address_model_user = "/cnvrg/" + args.Trained_Model_Name  #
address_model_cnvrg = args.default_model
rows_cnt = args.train_rows
sub1 = "train[:" + str(rows_cnt) + "]"
input_doc = datasets.load_dataset(
    "csv", data_files="/data/" + args.training_file, split=(str(sub1))
)
model = AutoModelForSeq2SeqLM.from_pretrained(address_model_cnvrg)
tokenizer = AutoTokenizer.from_pretrained("Tokenizer/")
encoder_max_length = int(args.encoder_max_length)
decoder_max_length = int(args.decoder_max_length)
weight_decay_factor = float(args.weight_decay_factor)
label_smooth_factor = float(args.label_smooth_factor)


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


data = datasets.load_dataset(
    "csv", data_files="/data/" + args.training_file, split=(str(sub1))
)
train_data_txt, validation_data_txt = data.train_test_split(test_size=0.1).values()
train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)
validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)
nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=1,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=4,  # demo
    per_device_eval_batch_size=4,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=weight_decay_factor,
    label_smoothing_factor=label_smooth_factor,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3
    #    push_to_hub=True,
    #    hub_token = 'api_kaIZZmGhrNmPTdaOrqUupOASRqtLoHqzyX',
    #    hub_model_id = "cnvrg/summarization"
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
evaluate_before = trainer.evaluate()  # type(evaluate_after)
trainer.train()
evaluate_after = trainer.evaluate()
model.save_pretrained(address_model_user)
print("Model is saved")  # input_doc['document'][7]
metrics_file_name = "eval_metrics.csv"
eval_metrics = pd.DataFrame(
    zip(
        list(evaluate_before.keys()),
        list(evaluate_before.values()),
        list(evaluate_after.values()),
    ),
    columns=["Metric", "Value_Before", "Value_After"],
)
eval_metrics.to_csv("/cnvrg/{}".format(metrics_file_name), index=False)
