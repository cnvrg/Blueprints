from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import pathlib
import os

scripts_dir = pathlib.Path(__file__).parent.resolve()
model_path = os.path.join(scripts_dir, 'trained_model_256_64_600')
tokenizer_path = os.path.join(scripts_dir, 'Tokenizer')
model_cnvrg = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
def predict(data):
    global model_cnvrg
    global tokenizer
    # address_model_cnvrg = '/cnvrg/Model/trained_model_256_64_600/'
    # model_cnvrg = AutoModelForSeq2SeqLM.from_pretrained(address_model_cnvrg)
    # tokenizer = AutoTokenizer.from_pretrained("/cnvrg/Model/Tokenizer/")
    encoder_max_length = 256

    def predict_1(text):
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
            return_tensors="pt",
            )
        input_ids = inputs.input_ids.to(model_cnvrg.device)
        attention_mask = inputs.attention_mask.to(model_cnvrg.device)
        min_length_1 = 0.07*len(text)
        outputs = model_cnvrg.generate(input_ids, attention_mask=attention_mask, max_length=500, min_length=round(min_length_1))
        outputs_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs_str

    response = predict_1(data['text'])
    return response
