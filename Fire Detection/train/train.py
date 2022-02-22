import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import SGD
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import argparse

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('--train_folder', action='store', dest='train_folder',
                    default='/data/fire-smoke-dataset/Train', required=True, help="""training data folder""")
parser.add_argument('--test_folder', action='store', dest='test_folder',
                    default='/data/fire-smoke-dataset/Test', required=True, help="""test data folder""")
parser.add_argument('--epochs', action='store', dest='epochs',
                    default=20, required=False, help="""epochs""")
parser.add_argument('--steps_per_epoch', action='store', dest='steps_per_epoch',
                    default=14, required=False)
parser.add_argument('--train_batch_size', action='store', dest='train_batch_size',
                    default=128, required=False)
parser.add_argument('--test_batch_size', action='store', dest='test_batch_size',
                    default=14, required=False)


parser.add_argument('--project_dir', action='store', dest='project_dir',
                    help="""--- For inner use of cnvrg.io ---""")
parser.add_argument('--output_dir', action='store', dest='output_dir',
                    help="""--- For inner use of cnvrg.io ---""")

args = parser.parse_args()
train_folder = args.train_folder
test_folder = args.test_folder
epochs = int(args.epochs)
steps_per_epoch = int(args.steps_per_epoch)
train_batch_size = int(args.train_batch_size)
test_batch_size = int(args.test_batch_size)

TRAINING_DIR = train_folder
training_datagen = ImageDataGenerator(rescale=1./255,
zoom_range=0.15,
horizontal_flip=True,
fill_mode='nearest')
VALIDATION_DIR = test_folder
validation_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = training_datagen.flow_from_directory(
TRAINING_DIR,
target_size=(224,224),
shuffle = True,
class_mode='categorical',
batch_size = train_batch_size)
validation_generator = validation_datagen.flow_from_directory(
VALIDATION_DIR,
target_size=(224,224),
class_mode='categorical',
shuffle = True,
batch_size= test_batch_size)


input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
  layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(
train_generator,
steps_per_epoch = steps_per_epoch,
epochs = 20,
validation_data = validation_generator,
validation_steps = 14)

for layer in model.layers[:249]:
  layer.trainable = False
for layer in model.layers[249:]:
  layer.trainable = True


model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(
train_generator,
steps_per_epoch = steps_per_epoch,
epochs = 10,
validation_data = validation_generator,
validation_steps = 14)

model.save('model.h5')