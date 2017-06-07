import sys
import os

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from keras.applications import VGG16
from keras.engine import Model

from implementations.support_scripts import image_processing
from keras import backend as K, Input
from keras import optimizers
from keras.layers import Conv2D, UpSampling2D, Lambda, Dense, Merge, merge, concatenate
from random import shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

b_size = 16
dir_name = "../small_dataset"
list_dir = os.listdir(dir_name)
shuffle(list_dir)
list_dir = list_dir
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
vgg16 = VGG16(weights=None, include_top=True)  # todo: change to imagenet
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


model = Model(inputs=[main_input, vgg16.input], output=last)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss="mean_squared_error")

model.summary()

start_from = 0
save_every_n_epoch = 10
n_epochs = 1000


g = image_processing.image_generator_parts(list_dir, b_size, im_size=(224, 224))

# for i in range(start_from // save_every_n_epoch, n_epochs // save_every_n_epoch):
    # model.fit_generator(g, steps_per_epoch=len(list_dir)//b_size, epochs=save_every_n_epoch)
    # model.save_weights("../weights/implementation1-" + str(i * save_every_n_epoch) + ".h5")

model.fit_generator(g, steps_per_epoch=len(list_dir) // b_size, epochs=n_epochs)

model.save_weights("../weights/imp7.h5")