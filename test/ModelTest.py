import unittest
from mlmodel.mnistexperiment import DnnOnnX
import onnxruntime
import os

class ModelTest (unittest.TestCase):

    def test_model(self):

        #####################    Test OnnX Molde
        onnxml = DnnOnnX(500, 10, 2)
        train_features, train_label, test_features, test_label, input_shape = onnxml.prep_data(28, 28)
        model = onnxml.build_model(input_shape)
        model_history, model = onnxml.train_model(model, train_features, train_label, test_features, test_label)
        onnxml.visualize_results(model_history)
        onnxml.convert_to_onnx(model, "onnxmodel.onnx")

        assert (os.path.isfile("onnxmodel.onnx"))


