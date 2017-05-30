import numpy as np
from skimage import io, color
from skimage.transform import resize
from random import shuffle
import os

import scipy.ndimage
# import matplotlib.pyplot as plt
# from scipy.misc import imread


from PIL import Image


# load data
def load_images(file):
    images = []

    rgb = io.imread("../small_dataset/" + file)
    img = Image.fromarray(rgb, 'RGB')

    img = img.resize((256, 256), Image.ANTIALIAS)

    img = img.convert(mode="RGB")  # ensure that image rgb
    rgb = np.array(img)

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


if __name__ == "__main__":
    # test
    list_dir = os.listdir("../../small_dataset")
    g = image_generator(list_dir, 2)
    a = next(g)

    print(a)
