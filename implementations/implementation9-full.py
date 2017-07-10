import pickle
import sys
import os

from skimage import color

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from implementations.support_scripts.image_processing import load_images_rgb, images_to_l
from implementations.support_scripts.image_tester import image_error_full_vgg

from implementations.support_scripts.common import h5_vgg_generator_let_there, image_check_with_vgg
# from implementations.support_scripts.metrics import root_mean_squared_error, mean_squared_error

from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K, Input
from keras import optimizers
from keras.layers import Conv2D, UpSampling2D, Lambda, Dense, Merge, merge, concatenate, regularizers, Add, add, \
    Conv2DTranspose, MaxPooling2D

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

b_size = 32

input_shape = (224, 224, 1)

# main network
main_input = Input(shape=input_shape, name='image_part_input')

x = Conv2D(64, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01), name="conv1")(main_input)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x1 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01), name="conv2")(x)
x = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01), name="conv3")(x1)
x = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01), name="conv4")(x)
x = add([x, x1])

x = Conv2D(128, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01))(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x1 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x1)
x = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = add([x, x1])

x = Conv2D(256, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01))(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x1)
x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = add([x, x1])

x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
main_output = Conv2D(256, (3, 3), padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(0.01))(x)

# VGG
vgg16 = VGG16(weights="imagenet", include_top=True)
vgg_output = Dense(256, activation='relu', name='predictions')(vgg16.layers[-2].output)

def repeat_output(input):
    shape = K.shape(x)
    return K.reshape(K.repeat(input, 28 * 28), (shape[0], 28, 28, 256))

vgg_output = Lambda(repeat_output)(vgg_output)

# freeze vgg16
for layer in vgg16.layers:
    layer.trainable = False

# concatenated net
merged = concatenate([vgg_output, main_output], axis=3)

last = Conv2D(128, (3, 3), padding="same")(merged)

last = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)

last = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(2, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)


def resize_image(x):
    return K.resize_images(x, 2, 2, "channels_last")


def unormalise(x):
    # outputs in range [0, 1] resized to range [-100, 100]
    return (x * 200) - 100


last = Lambda(resize_image)(last)
last = Lambda(unormalise)(last)


def custom_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3])


model = Model(inputs=[main_input, vgg16.input], output=last)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss=custom_mse)

model.summary()

start_from = 0
save_every_n_epoch = 1
n_epochs = 30
model.load_weights("../weights/implementation9-full-5.h5")

# start image downloader
#
# g = h5_vgg_generator_let_there(b_size, "../data/h5_224_train", None)
# gval = h5_vgg_generator_let_there(b_size, "../data/h5_224_validation", None)
#
#
# for i in range(start_from // save_every_n_epoch, n_epochs // save_every_n_epoch):
#     print("START", i * save_every_n_epoch, "/", n_epochs)
#     history = model.fit_generator(g, steps_per_epoch=100000//b_size, epochs=save_every_n_epoch,
#                                   validation_data=gval, validation_steps=(10000//b_size))
#     model.save_weights("../weights/implementation9-full-" + str(i * save_every_n_epoch) + ".h5")
#
#     # save sample images
#     image_check_with_vgg(model, 80, "imp9-full-" + str(i * save_every_n_epoch) + "-")
#
#     # save history
#     output = open('../history/imp9-full-{:0=4d}.pkl'.format(i * save_every_n_epoch), 'wb')
#     pickle.dump(history.history, output)
#     output.close()

# image_error_full_vgg(model, "imp9-full-100", b_size=b_size)

import numpy as np

abs_file_path = "../../subset100_000/validation"
image_list = os.listdir(abs_file_path)
num_of_images = len(image_list)

outputs = {layer.name : layer.output for layer in model.layers}

for batch_n in range(num_of_images // b_size):
    all_images_l = np.zeros((b_size, 224, 224, 1))
    all_images = np.zeros((b_size, 224, 224, 3))
    all_images_rgb = np.zeros((b_size, 224, 224, 3))
    for i in range(b_size):
        # get image
        image_rgb = load_images_rgb(abs_file_path, image_list[batch_n * b_size + i], size=(224, 224))  # image is of size 256x256
        image_lab = color.rgb2lab(image_rgb)
        image_l = images_to_l(image_lab)
        all_images_l[i, :, :, :] = image_l[:, :, np.newaxis]
        all_images[i, :, :, :] = image_lab
        all_images_rgb[i, :, :, :] = image_rgb

    all_vgg = np.zeros((num_of_images, 224, 224, 3))
    for i in range(b_size):
        all_vgg[i, :, :, :] = np.tile(all_images_l[i], (1, 1, 1, 3))

    # color_im = model.predict([all_images_l, all_vgg], batch_size=b_size)

    convout_ = K.function(model.inputs, [outputs["conv1"], outputs["conv2"], outputs["conv3"], outputs["conv4"]])

    net_layers = convout_([all_images_l, all_vgg])
    print(len(net_layers))


    exit()
