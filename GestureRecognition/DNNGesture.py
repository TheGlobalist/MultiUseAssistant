import tensorflow as tf
import tflearn
tf.disable_v2_behavior()
import numpy as np
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression


class DNN4GestureRecognition:
    def __init__(self):
        tf.reset_default_graph()
        convnet = input_data(shape=[None, 89, 100, 1], name='input')
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 128, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 256, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 256, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 128, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1000, activation='relu')
        convnet = dropout(convnet, 0.75)

        convnet = fully_connected(convnet, 3, activation='softmax')

        convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                             name='regression')

        self.__model = tflearn.DNN(convnet, tensorboard_verbose=0)

        # Load Saved Model
        self.__model.load("./GestureRecognition/TrainedModel/GestureRecogModel.tfl")

    def get_model(self):
        return self.__model

    def predict(self, norm_image):
        prediction = self.__model.predict([norm_image.reshape(89, 100, 1)])
        return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))