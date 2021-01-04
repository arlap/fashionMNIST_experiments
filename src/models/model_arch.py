from keras import Input, layers, Model


def dense_model():
    input_tensor = Input(shape=(28, 28, 1))
    x = layers.Flatten()(input_tensor)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)
    return model


def conv_model():
    input_tensor = Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      kernel_initializer='he_normal')(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)
    return model


def separableconv_model():
    # This won't work cause fashion-MNIST images in grayscale
    input_tensor = Input(shape=(28, 28, 1))
    x = layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu',
                               kernel_initializer='he_normal')(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.SeparableConv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.SeparableConv2D(128, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)
    return model
