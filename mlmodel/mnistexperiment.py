import onnx
import winmltools
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend
import matplotlib.pyplot as plt

class DnnOnnX:

    max_batch_size = 200
    number_of_classes = 10
    number_of_epocs = 20

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

        train_label = keras.utils.to_categorical(train_label, self.number_of_classes)
        test_label = keras.utils.to_categorical(test_label, self.number_of_classes)

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

        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return keras_model

    def train_model(self, model, train_features, train_label, test_features, test_label):

        model_history = model.fit(train_features, train_label,
                  batch_size=self.max_batch_size,
                  epochs=self.number_of_epocs,
                  verbose=1,
                  validation_data=(train_features, train_label))
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
        plt.plot(model_history.history['acc'], 'r', linewidth=2.0)
        plt.plot(model_history.history['val_acc'], 'b', linewidth=2.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=15)
        plt.xlabel('Epochs ', fontsize=13)
        plt.ylabel('Accuracy', fontsize=13)
        plt.title('Accuracy Curves', fontsize=13)
        plt.show()


    def convert_to_onnx(self, keras_model, onnx_model_name):

        ########  Conver to ONNX ############
        convert_model = winmltools.convert_keras(keras_model)

        winmltools.save_model(convert_model, onnx_model_name)