import os
import sys

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from implementations.support_scripts.metrics import root_mean_squared_error, mean_squared_error

from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K, Input
from keras import optimizers
from keras.layers import Conv2D, UpSampling2D, Lambda, Dense, Merge, merge, concatenate


def model():
    input_shape = (224, 224, 1)

    # main network
    main_input = Input(shape=input_shape, name='image_part_input')

    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(main_input)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)

    # middle layer
    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    main_output = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

    # VGG
    vgg16 = VGG16(weights="imagenet", include_top=True)
    vgg_output = Dense(256, activation='relu', name='predictions')(vgg16.layers[-2].output)

    def repeat_output(input):
        shape = K.shape(x)
        return K.reshape(K.repeat(input, 112 * 112), (shape[0], 112, 112, 256))

    vgg_output = Lambda(repeat_output)(vgg_output)

    # freeze vgg16
    for layer in vgg16.layers:
        layer.trainable = False

    # concatenated net
    merged = concatenate([vgg_output, main_output], axis=3)

    last = Conv2D(256, (3, 3), padding="same", activation="relu")(merged)
    last = Conv2D(128, (3, 3), padding="same", activation="relu")(last)

    last = UpSampling2D(size=(2, 2))(last)
    last = Conv2D(64, (3, 3), padding="same", activation="relu")(last)
    last = Conv2D(64, (3, 3), padding="same", activation="relu")(last)

    last = UpSampling2D(size=(2, 2))(last)
    last = Conv2D(32, (3, 3), padding="same", activation="relu")(last)
    last = Conv2D(2, (3, 3), padding="same", activation="sigmoid")(last)

    def resize_image(x):
        return K.resize_images(x, 2, 2, "channels_last")

    def unormalise(x):
        return (x * 200) - 100

    last = Lambda(resize_image)(last)
    last = Lambda(unormalise)(last)

    def custom_mse(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3])

    model = Model(inputs=[main_input, vgg16.input], output=last)
    opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss=custom_mse, metrics=[root_mean_squared_error, mean_squared_error])
    return model