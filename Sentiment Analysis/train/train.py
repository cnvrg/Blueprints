import keras
import tensorflow as tf
# DataFrame
import pandas as pd
# Matplot
import matplotlib.pyplot as plt
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# nltk
from nltk.stem import SnowballStemmer
# Word2vec
import gensim
# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import argparse

config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

print('gpuname: ', tf.config.experimental.list_physical_devices('GPU'))

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description="""Preprocessor""")

parser.add_argument('-f', '--input_filename', action='store', dest='input_filename', required=True,
                    help="""string. csv twitter labeled train data file""")

parser.add_argument('-o', '--output_token_file', action='store', dest='output_token_file', default='tokenizer.pickle',
                    required=False,
                    help="""string. filename for saving the tokenizer""")

parser.add_argument('-m', '--output_model_file', action='store', dest='output_model_file',
                    default='sentiment140_cpu_model.h5', required=False,
                    help="""string. filename for saving the model""")

parser.add_argument('-t', '--text_column', action='store', dest='text_column', default='text', required=False,
                    help="""string. name of text column""")

parser.add_argument('-s', '--label_column', action='store', dest='label_column', default='sentiment', required=False,
                    help="""string. name of label column""")

parser.add_argument('--epochs_val', action='store', dest='epochs_val', default=64, required=False,
                    help="""int. number of training epochs to run""")

parser.add_argument('--batch_size_val', action='store', dest='batch_size_val', default=256, required=False,
                    help="""int. size of each batch to train on""")

args = parser.parse_args()
output_token_file = args.output_token_file
output_model_file = args.output_model_file
text_column = args.text_column
label_column = args.label_column
epochs_val = int(args.epochs_val)
batch_size_val = int(args.batch_size_val)
input_filename = args.input_filename

# Settings
# DATASET
DATASET_COLUMNS = ["target", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = epochs_val
BATCH_SIZE = batch_size_val

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

df = pd.read_csv(input_filename, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}


def decode_sentiment(label):
    return decode_map[int(label)]


df.target = df.target.apply(lambda x: decode_sentiment(x))

# Pre-process dataset
stemmer = SnowballStemmer("english")


def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return " ".join(tokens)


df.text = df.text.apply(lambda x: preprocess(x))

# Split train and test
df_train, df_test = train_test_split(df, test_size=1 - TRAIN_SIZE, random_state=42)
documents = [_text.split() for _text in df_train.text]

# Word2vec
w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE,
                                            window=W2V_WINDOW,
                                            min_count=W2V_MIN_COUNT,
                                            workers=8)
w2v_model.build_vocab(documents)
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)
vocab_size = len(tokenizer.word_index) + 1

print("FINISHED WORD2VEC")

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)

print("FINISHED PADDING")
# Label encoder
labels = df_train.target.unique().tolist()
labels.append(NEUTRAL)

encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("FINISHED TRAINING")

# Embedding layer
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH,
                            trainable=False)

print("BUILDING MODEL")

# Build model
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Callbacks
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print()
print("ACCURACY:", score[1])
print("LOSS:", score[0])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save('/cnvrg/{}'.format(output_model_file))
