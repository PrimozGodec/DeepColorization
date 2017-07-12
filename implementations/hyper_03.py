import pickle
import sys
import os

from tensorflow.python.ops.image_ops_impl import ResizeMethod

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from implementations.support_scripts.image_tester import image_error_vgg

from implementations.support_scripts.common import h5_vgg_generator, image_check
from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K
from keras import optimizers
from keras.layers import concatenate, Conv2D, Lambda, UpSampling2D, BatchNormalization, Add, add

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

b_size = 8

# VGG
vgg16 = VGG16(weights="imagenet", include_top=True)

# build dict of layers
layer_dict = dict([(layer.name, layer) for layer in vgg16.layers])

# freeze vgg layers
for layer in vgg16.layers:
    layer.trainable = False

x = BatchNormalization()(layer_dict["block4_conv3"].output)
x = Conv2D(256, (1, 1), padding="same")(x)
x = UpSampling2D((2, 2))(x)

y = BatchNormalization()(layer_dict["block3_conv3"].output)
x = add([x, y])
x = Conv2D(128, (3, 3), padding="same")(x)
x = UpSampling2D((2, 2))(x)

y = BatchNormalization()(layer_dict["block2_conv2"].output)
x = add([x, y])
x = Conv2D(64, (3, 3), padding="same")(x)
x = UpSampling2D((2, 2))(x)

y = BatchNormalization()(layer_dict["block1_conv2"].output)
x = add([x, y])
x = Conv2D(3, (3, 3), padding="same")(x)

y = BatchNormalization()(vgg16.input)
x = add([x, y])
x = Conv2D(3, (3, 3), padding="same")(x)
output = Conv2D(2, (3, 3), padding="same")(x)

def unormalise(x):
    # outputs in range [0, 1] resized to range [-100, 100]
    return (x * 200) - 100

output = Lambda(unormalise)(output)

def custom_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3])


model = Model(inputs=vgg16.input, output=output)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss=custom_mse)

model.summary()

start_from = 0
save_every_n_epoch = 1
n_epochs = 150
# model.load_weights("../weights/hyper03-9.h5")

# start image downloader
ip = None

g = h5_vgg_generator(b_size, "../data/h5_224_train", ip)
gval = h5_vgg_generator(b_size, "../data/h5_224_validation", None)


for i in range(start_from // save_every_n_epoch, n_epochs // save_every_n_epoch):
    print("START", i * save_every_n_epoch, "/", n_epochs)
    history = model.fit_generator(g, steps_per_epoch=50000/b_size, epochs=save_every_n_epoch,
                                  validation_data=gval, validation_steps=(10000//b_size))
    model.save_weights("../weights/hyper03-" + str(i * save_every_n_epoch) + ".h5")

    # save sample images
    image_check(model, 80, "hyper03-" + str(i * save_every_n_epoch) + "-", b_size=b_size)

    # save history
    output = open('../history/hyper03-{:0=4d}.pkl'.format(i * save_every_n_epoch), 'wb')
    pickle.dump(history.history, output)
    output.close()

# image_error_vgg(model, "hyper03-test-100", b_size=b_size)