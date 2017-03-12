from typing import Tuple

from keras.layers import Activation, MaxPooling2D, Dropout, Convolution2D, Flatten, Dense
from keras.models import Sequential, Model


def create_vgg_like_model(input_shape: Tuple[int, int, int], output_units: int) -> Model:
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_units, activation='linear'))
    model.compile(optimizer='RMSprop', loss='mse', metrics=['mean_squared_error'])

    return model
