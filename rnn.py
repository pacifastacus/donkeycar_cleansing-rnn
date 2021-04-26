import os
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose


def rnn_lstm(seq_length=3, num_outputs=2, image_shape=(120,160,3)):

    
    img_seq_shape = (seq_length,) + image_shape   
    img_in = Input(batch_shape = img_seq_shape, name='img_in')
    drop_out = 0.3

    x = Sequential()
    x.add(TD(Cropping2D(cropping=((40,0), (0,0))), input_shape=img_seq_shape )) #trim 60 pixels off top
    x.add(TD(BatchNormalization()))
    x.add(TD(Convolution2D(24, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (3,3), strides=(2,2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (3,3), strides=(1,1), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(MaxPooling2D(pool_size=(2, 2))))
    x.add(TD(Flatten(name='flattened')))
    x.add(TD(Dense(100, activation='relu')))
    x.add(TD(Dropout(drop_out)))
      
    x.add(LSTM(128, return_sequences=True, name="LSTM_seq"))
    x.add(Dropout(.1))
    x.add(LSTM(128, return_sequences=False, name="LSTM_out"))
    x.add(Dropout(.1))
    x.add(Dense(128, activation='relu'))
    x.add(Dropout(.1))
    x.add(Dense(64, activation='relu'))
    x.add(Dense(10, activation='relu'))
    x.add(Dense(num_outputs, activation='linear', name='model_outputs'))
    
    return x


# def compile(optim="rmsprop", seq_len=3, num_outp=2, im_shape=(120,160,3)):
# 	model = rnn_lstm(seq_length=seq_len, num_outputs=num_outp, image_shape=im_shape)
#   model.compile(optimizer=optim, loss='mse')


