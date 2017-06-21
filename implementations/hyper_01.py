import pickle
import sys
import os

from tensorflow.python.ops.image_ops_impl import ResizeMethod

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from implementations.support_scripts.common import whole_image_check, h5_small_vgg_generator, \
    whole_image_check_overlapping, h5_vgg_generator
from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K, Input
from keras import optimizers
from keras.layers import concatenate, Conv2D, Lambda, UpSampling2D, MaxPooling2D

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

b_size = 8

# VGG
vgg16 = VGG16(weights="imagenet", include_top=True)

# build dict of layers
layer_dict = dict([(layer.name, layer) for layer in vgg16.layers])

# freeze vgg layers
for layer in vgg16.layers:
    layer.trainable = False

def resize(x):
    return tf.image.resize_images(x, (224, 224), method=ResizeMethod.BILINEAR)


# block2_conv1 = Lambda(resize)(layer_dict["block2_conv1"].output)
block2_conv2 = Lambda(resize)(layer_dict["block2_conv2"].output)

# block3_conv1 = Lambda(resize)(layer_dict["block3_conv1"].output)
# block3_conv2 = Lambda(resize)(layer_dict["block3_conv2"].output)
block3_conv3 = Lambda(resize)(layer_dict["block3_conv3"].output)

# block4_conv1 = Lambda(resize)(layer_dict["block4_conv1"].output)
# block4_conv2 = Lambda(resize)(layer_dict["block4_conv2"].output)
block4_conv3 = Lambda(resize)(layer_dict["block4_conv3"].output)

# block5_conv1 = Lambda(resize)(layer_dict["block5_conv1"].output)
# block5_conv2 = Lambda(resize)(layer_dict["block5_conv2"].output)
block5_conv3 = Lambda(resize)(layer_dict["block5_conv3"].output)



def repeat_output(input):
    shape = K.shape(input)
    return K.reshape(K.repeat(input, 224 * 224), (shape[0], 224, 224, 4096))


# fc1 = Lambda(repeat_output)(layer_dict["fc1"].output)
# fc2 = Lambda(repeat_output)(layer_dict["fc2"].output)

hypercolumns = concatenate([layer_dict["block1_conv2"].output, block2_conv2,
                            block3_conv3, block4_conv3,block5_conv3], axis=3)

# hypercolumns = concatenate([fc1, fc2], axis=3)

output = Conv2D(2, (3, 3), padding="same", activation="relu")(hypercolumns)
output = UpSampling2D((4, 4))(output)

def custom_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3])


model = Model(inputs=vgg16.input, output=output)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss=custom_mse)

model.summary()

start_from = 0
save_every_n_epoch = 5
n_epochs = 10000
# model.load_weights("../weights/implementation7d-reg-55.h5")

# start image downloader
ip = None

g = h5_vgg_generator(b_size, "../h5_data_224", ip)
gval = h5_vgg_generator(b_size, "../h5_data_224_validate", None)


for i in range(start_from // save_every_n_epoch, n_epochs // save_every_n_epoch):
    print("START", i * save_every_n_epoch, "/", n_epochs)
    history = model.fit_generator(g, steps_per_epoch=60000/b_size, epochs=save_every_n_epoch,
                                  validation_data=gval, validation_steps=(1024//b_size))
    model.save_weights("../weights/hyper01-" + str(i * save_every_n_epoch) + ".h5")

    # save sample images
    whole_image_check_overlapping(model, 40, "hyper01-" + str(i * save_every_n_epoch) + "-")

    # save history
    output = open('../history/hyper01-{:0=4d}.pkl'.format(i * save_every_n_epoch), 'wb')
    pickle.dump(history.history, output)
    output.close()

