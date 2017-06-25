import pickle
import sys
import os

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from implementations.support_scripts.metrics import root_mean_squared_error, mean_squared_error

from implementations.support_scripts.common import whole_image_check, h5_small_vgg_generator, \
    whole_image_check_overlapping, h5_vgg_generator, image_check
from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K, Input
from keras import optimizers
from keras.layers import Conv2D, UpSampling2D, Lambda, Dense, Merge, merge, concatenate, regularizers

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

b_size = 32

# VGG
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# freeze vgg16
for layer in vgg16.layers:
    layer.trainable = False

last = UpSampling2D(size=(2, 2))(vgg16.output)
last = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)

last = UpSampling2D(size=(2, 2))(last)
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


model = Model(inputs=vgg16.input, output=last)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss=custom_mse, metrics=[root_mean_squared_error, mean_squared_error])

model.summary()

start_from = 0
save_every_n_epoch = 3
n_epochs = 10000
# model.load_weights("../weights/vgg1-0.h5")

# start image downloader
ip = None

g = h5_vgg_generator(b_size, "../data/h5_224_train", ip)
gval = h5_vgg_generator(b_size, "../data/h5_224_validation", None)


for i in range(start_from // save_every_n_epoch, n_epochs // save_every_n_epoch):
    print("START", i * save_every_n_epoch, "/", n_epochs)
    history = model.fit_generator(g, steps_per_epoch=100000//b_size, epochs=save_every_n_epoch,
                                  validation_data=gval, validation_steps=(10000//b_size))
    model.save_weights("../weights/vgg1-" + str(i * save_every_n_epoch) + ".h5")

    # save sample images
    image_check(model, 80, "vgg1-" + str(i * save_every_n_epoch) + "-")

    # save history
    output = open('../history/vgg1-{:0=4d}.pkl'.format(i * save_every_n_epoch), 'wb')
    pickle.dump(history.history, output)
    output.close()

