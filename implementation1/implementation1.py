import os

import scipy.ndimage
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Conv2DTranspose
from keras import backend as K
from keras import optimizers

import numpy as np
from skimage import io, color
from skimage.transform import resize
from random import shuffle

import pickle


os.environ["CUDA_VISIBLE_DEVICES"]="1"


def custom_softmax(x):
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


# load data
def load_images(file):
    images = []

    rgb = io.imread("../small_dataset/" + file)
    rgb = resize(rgb, (256, 256), mode='reflect')
    if len(rgb.shape) == 3 and (rgb.shape[2]) == 3:  # avoid black and white photos
        return color.rgb2lab(rgb)
    else:
        print(file)


def images_to_l(image):
    return image[:, :, 0]


def images_to_ab(image):
    return image[:, :, 1:3]


def lab2rgb(l, ab):
    return color.lab2rgb(np.concatenate((l, ab), axis=2))


def save_rgb(file_name, im):
    io.imsave(file_name, im)

# cielab properties - according to responses
# from https://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space
l_range = {"min": 0, "max": 100}
a_range = {"min": -87, "max": 99}
b_range = {"min": -108, "max": 95}
n_bins = 20
a_bin_len = a_range["max"] - a_range["min"]
b_bin_len = b_range["max"] - b_range["min"]


def ab_to_histogram(ab):
    indices = (ab[:, :, 0] - a_range["min"]) // a_bin_len + (ab[:, :, 1] - b_range["min"]) // b_bin_len
    one_hot = np.zeros((ab.shape) + n_bins ** 2)
    one_hot[np.arange(ab.shape[0]), np.arange(ab.shape[1]),  indices] = 1
    return one_hot


def gaussian_filter(data, sigma=5):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            l = scipy.ndimage.filters.gaussian_filter(np.reshape(data[i, j, :], (n_bins, n_bins)), sigma=sigma)
            data[i, j, :] = l
    return data




def image_generator(image_dir, batch_size):
    n = 0
    while True:
        batch_im_names = image_dir[n:n+batch_size]
        print(len(batch_im_names))

        batch_imputs = np.zeros((batch_size, 256, 256, 1))
        batch_targets = np.zeros((batch_size, 256, 256, 2))
        for i, image in enumerate(batch_im_names):
            im = load_images(image)
            batch_imputs[i] = images_to_l(im)[:, :, np.newaxis]
            batch_targets[i] = images_to_ab(im)
        yield (batch_imputs, batch_targets)
        n += batch_size
        if n + batch_size > len(image_dir):
            n = 0
            shuffle(image_dir)


b_size = 128
list_dir = os.listdir("../small_dataset")

model = Sequential()

# conv1_1
model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(256, 256, 1)))
# conv1_2
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'))

# conv2_1
model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
# conv2_2
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu'))

# conv3_1
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# conv3_2
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# conv3_3
model.add(Conv2D(256, (3, 3), padding="same", strides=(2, 2), activation='relu'))

# conv4_1
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv4_2
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv4_3
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))

# conv5_1
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv5_2
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv5_3
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))

# conv6_1
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv6_2
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv6_3
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))

# conv7_1
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# conv7_2
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# conv7_3
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))

# conv8_1
model.add(Conv2DTranspose(128, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# conv8_2
model.add(Conv2DTranspose(128, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# conv8_3
model.add(Conv2DTranspose(64, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# conv8_4
model.add(Conv2D(2, (3, 3), padding="same",  activation='relu'))

# multidimensional softmax
# todo: try the way to use default function - with axis
model.add(Activation(custom_softmax))


sgd = optimizers.SGD(lr=1, momentum=0.0, decay=0, nesterov=False)
model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=['accuracy'])

model.summary()

model.fit_generator(image_generator(list_dir, b_size),
                    steps_per_epoch=len(list_dir)//b_size, epochs=2)

model.save_weights('implementation1_1.h5')
# model.load_weights('implementation1_1.h5')
#
# prediction = model.predict(images_to_l(load_images(list_dir[0])))
#
# with open('pred.pickle', 'wb') as handle:
#     pickle.dump(prediction, handle)