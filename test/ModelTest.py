import unittest
import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
print(CURR_DIR)

import sys
sys.path.append('../mlmodel')

from mlmodel.mnistexperiment import DnnOnnX
import onnxruntime
import os

class ModelTest (unittest.TestCase):


    
    
    def test_model(self):

        #####################    Test OnnX Molde
        onnxml = DnnOnnX(500, 10, 20)
        train_features, train_label, test_features, test_label, input_shape = onnxml.prep_data(28, 28)
        model = onnxml.build_model(input_shape)
        model_history, model = onnxml.train_model(model, train_features, train_label, test_features, test_label)
        onnxml.visualize_results(model_history)
        onnxml.convert_to_onnx(model, "onnxmodel.onnx")
        assert (os.path.isfile("onnxmodel.onnx"))
        
        
          
    """[summary] 
    
    def test_model_quantizaiton(self):
        
        dnn_model = 'dnn_saved_model'
        new_tflite_model = 'dnn_tflite_saved_model'
        new_quantized_model = 'dnn_quantized_model'
        quant_model = DnnOnnX(500, 10, 3)
        train_features, train_label, test_features, test_label, input_shape = quant_model.prep_data(28, 28)
        model = quant_model.build_model(input_shape)
        model_history, model = quant_model.train_model(model, train_features, train_label, test_features, test_label)
        quant_model.visualize_results(model_history)
        
        quant_model.save_model(model, dnn_model)
        
        
        quant_model.post_training_convert_to_tf_lite(dnn_model, new_tflite_model)
        
        new_quant_model =  quant_model.post_training_apply_quantization (dnn_model,new_quantized_model)
        quant_model.test_model(new_quantized_model+"/tf_quan_model.tflite",test_features, test_label)
      

      
    def test_train_quantized_model (self):
        
        quantml = DnnOnnX(500, 10, 2)
        train_features, train_label, test_features, test_label, input_shape = quantml.prep_data(28, 28)
        model = quantml.build_model(input_shape)
        model_history, model = quantml.train_model(model, train_features, train_label, test_features, test_label)     
        
        #train quantized
        quantml.quantized_training(model, train_features, train_label, test_features, test_label)
        
     
      """
        
            


