from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tf.keras.layers import *
from tf.keras import *


def get_pipeline(x_traincnn, y_train_lb, x_testcnn, y_test_lb):
    CNN_model = keras.Sequential()

    #Build first layer
    CNN_model.add(keras.Conv1D(16, 5,padding='same',
                     input_shape=(40, 1), activation='relu'))

    #Build second layer
    CNN_model.add(keras.Conv1D(32, 5,padding='same',activation='relu'))

    #Build third layer
    CNN_model.add(keras.Conv1D(64, 5,padding='same',activation='relu'))

    #Build forth layer
    CNN_model.add(keras.Conv1D(128, 5,padding='same',activation='relu'))

    #Add dropout
    CNN_model.add(Dropout(0.1))

    #Flatten
    CNN_model.add(Flatten())

    CNN_model.add(Dense(128, activation ='relu'))
    CNN_model.add(Dropout(0.1))
    CNN_model.add(Dense(64, activation ='relu'))
    CNN_model.add(Dense(8, activation='softmax'))

    CNN_model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'Adam',
                      metrics = ['accuracy'])

    cnn_results = CNN_model.fit(x_traincnn, y_train_lb,
                  batch_size = 64,
                  epochs = 25,
                  verbose = 1,
                  validation_data = (x_testcnn, y_test_lb))

    return cnn_results