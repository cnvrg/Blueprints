import os
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pathlib

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
scripts_dir = pathlib.Path(__file__).parent.resolve()
model_path = os.path.join(scripts_dir, 'sentiment140_cpu_model.h5')
tokenizer_path = os.path.join(scripts_dir, 'sentiment140_tokenizer.pickle')
print(model_path)
sentiment140_model = keras.models.load_model(model_path, compile=False)
print("loaded model")
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

sentiment_label = ['positive', 'negative']

def predict(data):
    sentiment_label = ['negative', 'positive']
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([data['text']]), maxlen=300)
    # Predict
    score = sentiment140_model.predict([x_test])[0]
    # Decode sentiment
    label = sentiment_label[int(score.round().item())]
    return {"label": label, "score": float(score)}

