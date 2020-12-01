from flask import Flask
from flask import request
import numpy as np
import cv2

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mlmodel.score  import init, run

app = Flask(__name__)

init()

@app.route("/")
def intro():
    return "welcome to ONNX operationalize ML models"

@app.route("/api/classify_digit", methods=['POST'])
def classify():
    #print (request.get_data())
    #print (len(request.get_data()))
    #print (type (request.get_data()))
    input_img = np.fromstring(request.get_data(), np.uint8)
    print (type (input_img))
    print (input_img)

    img = cv2.imdecode(input_img, cv2.IMREAD_GRAYSCALE)
    classify_response = "".join(map(str, run(img)))

    return (classify_response)

if __name__ == 'main':
    app.run(debug = True, host='0.0.0.0', port=8000)

