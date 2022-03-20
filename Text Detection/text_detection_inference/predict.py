import os
import cv2
import argparse
import base64
import numpy as np
import magic
import pathlib
import sys
scripts_dir = pathlib.Path(__file__).parent.resolve()
easyocr_dir = os.path.join(scripts_dir, 'easyocr')
sys.path.append(easyocr_dir)
#os.environ['EASYOCR_MODULE_PATH'] = os.path.join(scripts_dir,'model')
import easyocr


language = ['en']
if "language" in os.environ:
    language = os.environ["language"].split(",")
reader = easyocr.Reader(language, model_storage_directory=os.path.join(scripts_dir,'model'), download_enabled=False)

def predict(data):
    output = {}

    for image_number, image_data in enumerate(data["img"]):
        output[image_number + 1] = []
        decoded = base64.b64decode(image_data)
        file_ext = magic.from_buffer(decoded, mime=True).split("/")[-1]
        savepath = f"img.{file_ext}"
        nparr = np.fromstring(decoded, np.uint8)
        img_dec = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(savepath, img_dec)
        result = reader.readtext(
            savepath,
            rotation_info=[90, 180, 270],
            contrast_ths=0.6,
            adjust_contrast=0.99,
        )
        for i in range(len(result)):
            x = result[i][0][0][0]
            y = result[i][0][0][1]
            w = result[i][0][1][0] - result[i][0][0][0]
            h = result[i][0][2][1] - result[i][0][1][1]
            response = {
                "text": result[i][1],
                "bounding_box": [int((x + w) / 2), int((y + h) / 2), int(w), int(h)],
                "confidence": result[i][-1],
            }
            output[image_number + 1].append(response)
    return output