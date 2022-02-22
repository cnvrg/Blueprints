import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
import base64
import pathlib
import os


scripts_dir = pathlib.Path(__file__).parent.resolve()
model_path = os.path.join(scripts_dir, 'model.h5')
model = tf.keras.models.load_model(model_path)

def predict(data):
    predictions = []
    for i in data['img']:
        decoded = base64.b64decode(i['base_64_content'])
        nparr = np.fromstring(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        # img = cv2.imread(data)
        img = cv2.resize(img, (224,224))
        img = image.img_to_array(img)
        img_array = np.expand_dims(img, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        # prediction = [{'label': 'fire', 'score': float(probabilities[0])},
        #               {'label': 'natural', 'score': float(probabilities[1])}]
        pred = np.argmax(probabilities)
        prediction = {'name': i['name'],
                      'label': int(pred),
                      'score': float(probabilities[pred])}
        predictions.append(prediction)
    return {'prediction': predictions}
    