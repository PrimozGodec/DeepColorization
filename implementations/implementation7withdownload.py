import pickle
import sys
import os

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])
from implementations.support_scripts.image2h5_packer import ImagePacker

from implementations.support_scripts.common import test_whole_image, h5_small_vgg_generator
from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K, Input
from keras import optimizers
from keras.layers import Conv2D, UpSampling2D, Lambda, Dense, Merge, merge, concatenate


os.environ["CUDA_VISIBLE_DEVICES"] = "5"

b_size = 32

num_classes = 40
input_shape = (32, 32, 1)

# main network
main_input = Input(shape=input_shape, name='image_part_input')

x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(main_input)
x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
main_output = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

# VGG
vgg16 = VGG16(weights="imagenet", include_top=True)
vgg_output = Dense(256, activation='softmax', name='predictions')(vgg16.layers[-2].output)

def repeat_output(input):
    shape = K.shape(x)
    return K.reshape(K.repeat(input, 4 * 4), (shape[0], 4, 4, 256))

vgg_output = Lambda(repeat_output)(vgg_output)

# freeze vgg16
for layer in vgg16.layers:
    layer.trainable = False

# concatenated net
merged = concatenate([vgg_output, main_output], axis=3)

last = Conv2D(128, (3, 3), padding="same")(merged)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(64, (3, 3), padding="same", activation="sigmoid")(last)
last = Conv2D(64, (3, 3), padding="same", activation="sigmoid")(last)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(32, (3, 3), padding="same", activation="sigmoid")(last)
last = Conv2D(2, (3, 3), padding="same", activation="sigmoid")(last)


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
save_every_n_epoch = 500
n_epochs = 10000
# model.load_weights("../weights/implementation7d-5400.h5")

# start image downloader
ip = ImagePacker("../small_dataset", "../h5_data",  "imp7l", num_images=1024, num_files=2)
ip.start()

g = h5_small_vgg_generator(b_size, "../h5_data", ip)
gval = h5_small_vgg_generator(b_size, "../h5_validate", None)

for i in range(start_from // save_every_n_epoch, n_epochs // save_every_n_epoch):
    print("START", i * save_every_n_epoch, "/", n_epochs)
    history = model.fit_generator(g, steps_per_epoch=1024/b_size, epochs=save_every_n_epoch,
                                  validation_data=gval, validation_steps=128/b_size)
    model.save_weights("../weights/implementation7d-" + str(i * save_every_n_epoch) + ".h5")

    # save sample images
    test_whole_image(model, 20, "imp7d-" + str(i * save_every_n_epoch) + "-")

    # save history
    output = open('../history/imp7d-{:0=4d}.pkl'.format(i * save_every_n_epoch), 'wb')
    pickle.dump(history.history, output)
    output.close()

