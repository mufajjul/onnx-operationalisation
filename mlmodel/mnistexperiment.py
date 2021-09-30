from os import makedirs
import tensorflow as tf
from tensorflow.python import tf2
import onnx
import winmltools
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
import datetime
import os.path
import numpy as np

class DnnOnnX:

    max_batch_size = 200
    number_of_classes = 10
    number_of_epocs = 200
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def __init__(self, batch_size, total_classes, epochs_size):

        self.max_batch_size = batch_size
        self.number_of_classes = total_classes
        self.number_of_epocs = epochs_size

    def prep_data(self, image_row_size, image_column_size):

        input_shape= None

        (train_features, train_label), (test_features, test_label) = mnist.load_data()

        train_features = train_features.astype('float32')
        train_features/=255
        test_features = test_features.astype('float32')
        test_features/=255

        if backend.image_data_format() == 'channels_first':
            train_features = train_features.reshape(train_features.shape[0], 1, image_row_size, image_column_size)
            test_features = test_features.reshape(test_features.shape[0], 1, image_row_size, image_column_size)
            input_shape = (1, image_row_size, image_column_size)
        else:
            train_features = train_features.reshape(train_features.shape[0], image_row_size, image_column_size, 1)
            test_features =test_features.reshape(test_features.shape[0], image_row_size, image_column_size, 1)
            input_shape = (image_row_size, image_column_size,1)

        train_label = tf.keras.utils.to_categorical(train_label, self.number_of_classes)
        test_label = tf.keras.utils.to_categorical(test_label, self.number_of_classes)

        return train_features, train_label, test_features, test_label, input_shape


    def build_model(self, input_shape):

        keras_model = Sequential()
        keras_model.add(Conv2D(16, strides= (1,1), padding= 'valid', kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
        keras_model.add(Conv2D(32, (3, 3), activation='relu'))
        keras_model.add(MaxPooling2D(pool_size=(2, 2)))
        keras_model.add(Dropout(0.25))
        keras_model.add(Conv2D(64, (3, 3), activation='relu'))
        keras_model.add(MaxPooling2D(pool_size=(2, 2)))
        keras_model.add(Dropout(0.25))
        keras_model.add(Flatten())
        keras_model.add(Dense(128, activation='relu'))
        keras_model.add(Dropout(0.5))
        keras_model.add(Dense(self.number_of_classes, activation='softmax'))
        
        
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        #rmse_prop = tensorflow.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0)
        #ada_grad = tensorflow.keras.optimizers.Adagrad(learning_rate=0.001)
        #ada_delta = tf.keras.optimizers.Adadelta(learning_rate=0.01)
        #SGD_optimizer = tensorflow.optimizers.SGD(learning_rate=0.01, momentum=0.7)
#        keras_model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
#                      optimizer=tensorflow.keras.optimizers.Adadelta(),
#                      metrics=['accuracy'])
        #nAdam_optimizer = tf.optimizers.Nadam(learning_rate=0.001)
        

        keras_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=adam_optimizer,
                      metrics=['accuracy'])



        return keras_model

    def train_model(self, model, train_features, train_label, test_features, test_label):
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)        

        model_history = model.fit(train_features, train_label,
                  batch_size=self.max_batch_size,
                  epochs=self.number_of_epocs,
                  verbose=1,
                  validation_data=(train_features, train_label), callbacks=[tensorboard_callback])
        score = model.evaluate(test_features, test_label, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return model_history, model

    def visualize_results(self, model_history):

        # Plot the Loss Curves
        plt.figure(figsize=[8, 6])
        plt.plot(model_history.history['loss'], 'r', linewidth=3.0)
        plt.plot(model_history.history['val_loss'], 'b', linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves', fontsize=16)
        plt.show()

        # Plot the Accuracy Curves
        plt.figure(figsize=[10, 8])
        plt.plot(model_history.history['accuracy'], 'r', linewidth=2.0)
        plt.plot(model_history.history['val_accuracy'], 'b', linewidth=2.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=15)
        plt.xlabel('Epochs ', fontsize=13)
        plt.ylabel('Accuracy', fontsize=13)
        plt.title('Accuracy Curves', fontsize=13)
        plt.show()


    def quantized_training(self, model, train_features, train_label, test_features, test_label):
        
        
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model (model)
        
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        q_aware_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=adam_optimizer,
                      metrics=['accuracy'])
        
               
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)        

        model_history = q_aware_model.fit(train_features, train_label,
                  batch_size=self.max_batch_size,
                  epochs=self.number_of_epocs,
                  verbose=1,
                  validation_data=(train_features, train_label), callbacks=[tensorboard_callback])
        
        score = q_aware_model.evaluate(test_features, test_label, verbose=0)
        
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return model_history, q_aware_model
             

    def convert_to_onnx(self, keras_model, onnx_model_name):

        ########  Conver to ONNX ############
        convert_model = winmltools.convert_keras(keras_model)
        winmltools.save_model(convert_model, onnx_model_name)
        
    def save_model (self, keras_model, save_model_path):
        keras_model.save (save_model_path)
    
    
    # convert to TF lite model with out quantization  
    def post_training_convert_to_tf_lite(self, saved_model_path, new_model_save_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        tflite_model = converter.convert()
        #tflite_model.save(new_model_save_path)
        
        if (not (os.path.isdir(new_model_save_path))):
            os.makedirs(new_model_save_path)
        
        with open (new_model_save_path+"/tf_model.tflite", 'wb') as f:
            f.write (tflite_model)
            
        return tflite_model
        
        
    # apply quantization to tf lite model 
    #apply_quantized tf lite modle     
    def post_training_apply_quantization (self, tf_lite_model_path, new_tf_lite_quantized_model):
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_lite_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        #tflite_quant_model.save(new_tf_lite_quantized_model)
        
        if (not (os.path.isdir(new_tf_lite_quantized_model))):
            os.makedirs (new_tf_lite_quantized_model)
        
        with open (new_tf_lite_quantized_model+"/tf_quan_model.tflite", 'wb') as f:
            f.write (tflite_quant_model)
            
        return tflite_quant_model
    
            
    def test_model (self, model_path, test_features, test_label):        
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        
        print ('input shape', input_shape)
        print (test_features.shape)
        print (test_features[0:1:].shape)
        
        #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_features[0:1:])

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        print (test_label.shape)
        print (test_label[0])
        
        #score = model.evaluate(test_features, test_label, verbose=0)
        #print('Test loss:', score[0])
        #print('Test accuracy:', score[1])
        
