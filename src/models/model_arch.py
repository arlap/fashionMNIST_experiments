from keras.models import Sequential
from keras import Input, layers, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, SeparableConv2D, Flatten


def dense_model():
    input_tensor = Input(shape=(28, 28))
    x = layers.Flatten()(input_tensor)
    x = layers.Dense(128, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)
    return model


def conv_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_normal',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    return model


def separableconv_model():
    model = Sequential()
    model.add(SeparableConv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              kernel_initializer='he_normal',
                              input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(SeparableConv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(SeparableConv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    return model
