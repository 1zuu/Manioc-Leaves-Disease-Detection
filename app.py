import os
import numpy as np
import cv2 as cv
from flask import Flask, jsonify, request

# from util import *
from variables import *
from cnn import ManiocDiseaseDetection

app = Flask(__name__)

model = ManiocDiseaseDetection()
model.run()

def preprocessing_function(img):
    img = (img - 127.5) / 127.5
    return img

def preprocess_image(image):
    if image.shape[-1] == 1:
        return False
    else:
        image = cv.resize(image, target_size, cv.INTER_AREA)
        return image

@app.route("/prediction", methods=["POST"])
def predict():

      dogimagefile = request.files['image'].read()
      dogimage = np.fromstring(dogimagefile, np.uint8)
      dogimage = cv.imdecode(dogimage,cv.IMREAD_COLOR) 
      dogimage = preprocess_image(dogimage)
      if dogimage.any():
        dogimage = preprocessing_function(dogimage)
        dogimage = dogimage.astype(np.float32)
        disease_sentiment = model.Inference(dogimage)
        print(disease_sentiment)
        response = {
                    "disease_sentiment": disease_sentiment
                   }
        return jsonify(response)

      else:
        return "Please Insert RGB image of your DOG !"


if __name__ == "__main__": 
    app.run(debug=True, host=host, port= port, threaded=False, use_reloader=True)