import json
import numpy
from numpyencoder import NumpyEncoder
import onnxruntime
import cv2

def init():
    global onnx_session
    onnx_session = onnxruntime.InferenceSession('model/onnxmodel.onnx')

def run(raw_data):
    try:

        newimg = cv2.resize(raw_data, (int(28), int(28)))
        reshaped_img = newimg.reshape(1, 28, 28, 1)

        classify_output = onnx_session.run(None, {onnx_session.get_inputs()[0].name:reshaped_img.astype('float32')})
        print (type(classify_output))
        print (classify_output)
        return json.dumps({"prediction":classify_output}, cls=NumpyEncoder)

    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})



