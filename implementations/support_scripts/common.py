from random import shuffle

import numpy as np
import scipy.misc
from os import listdir
import os

from os.path import isfile, join

import inspect

from keras.utils import HDF5Matrix
from skimage import color
import keras.backend as K


"""
This script contains functions that are common to all the implementations used
"""
from implementations.support_scripts.image_processing import image_generator_hist, histogram_to_ab, \
    histogram_to_ab_separate, load_images, images_to_l, resize_image


def make_prediction_sample(model, batch_size, name, generator=None):
    if generator is None:
        generator = image_generator_hist(None, "../test_set", batch_size)  # just generate batch of 10 images
    images_l = next(generator)[0]
    predictions_ab = model.predict(images_l)

    for i in range(images_l.shape[0]):
        # concatenate l and ab for each image
        if images_l[i, :, :, :].shape[2] == 40:  # if only 40 bins there is separate implementation
            im = np.concatenate((images_l[i, :, :, :], histogram_to_ab_separate(predictions_ab[i, :, :, :])), axis=2)
        else:
            im = np.concatenate((images_l[i, :, :, :], histogram_to_ab(predictions_ab[i, :, :, :])), axis=2)
        im_rgb = color.lab2rgb(im)
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save('../result_images/' + name + str(i) + '.jpg')


def make_prediction_sample_part(model, batch_size, name, generator=None):
    if generator is None:
        generator = image_generator_hist(None, "../test_set", batch_size)  # just generate batch of 10 images
    images_l = next(generator)[0]
    predictions_ab = model.predict(images_l)

    for i in range(images_l[0].shape[0]):
        # concatenate l and ab for each image
        im = np.concatenate((images_l[0][i, :, :, :], predictions_ab[i, :, :, :]), axis=2)

        im_rgb = color.lab2rgb(im)
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save('../result_images/' + name + str(i) + '.jpg')


def test_whole_image(model, num_of_images, name):

    # find directory
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
    rel_path = "../../test_set"
    abs_file_path = os.path.join(script_dir, rel_path)
    image_list = os.listdir(abs_file_path)

    # repeat for each image
    # lets take first n images
    for i in range(num_of_images):
        # get image
        image_lab = load_images(abs_file_path, image_list[i])  # image is of size 256x256
        image_l = images_to_l(image_lab)

        h, w = image_l.shape

        # split images to list of images
        slices_dim = 256//32
        slices = np.zeros((slices_dim * slices_dim, 32, 32, 1))
        for a in range(slices_dim):
            for b in range(slices_dim):
                slices[a * slices_dim + b] = image_l[a * 32 : (a + 1) * 32, b * 32 : (b + 1) * 32, np.newaxis]

        # lover originals dimension to 224x224 to feed vgg and increase dim
        image_l_224_b = resize_image(image_l, (224, 224))
        image_l_224 = np.repeat(image_l_224_b[:, :, np.newaxis], 3, axis=2).astype(float)


        # append together booth lists
        input_data = [slices, np.array([image_l_224,] * slices_dim ** 2)]

        # predict
        predictions_ab = model.predict(input_data)

        # reshape back
        original_size_im = np.zeros((h, w, 2))
        for n in range(predictions_ab.shape[0]):
            a, b = n // slices_dim * 32, n % slices_dim * 32
            original_size_im[a:a+32, b:b+32, :] = predictions_ab[n, :, :]

        # to rgb
        color_im = np.concatenate((image_l[:, :, np.newaxis], original_size_im), axis=2)
        # color_im = np.concatenate(((np.ones(image_l.shape) * 50)[:, :, np.newaxis], original_size_im), axis=2)
        im_rgb = color.lab2rgb(color_im)

        # save
        abs_svave_path = os.path.join(script_dir, '../../result_images/')
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + name + image_list[i])


def data_to_onehot(data):
    t = K.one_hot(K.round(data), 400)
    tf_session = K.get_session()
    return t.eval(session=tf_session)

class H5Choose:

    def __init__(self, dir):
        self.dir = dir
        self.used = []

    def all_files(self):
        return [f for f in listdir(self.dir) if f.endswith("h5")]

    def pick_next(self, downloader):
        only_files = self.all_files()
        not_used = sorted(list(set(only_files) - set(self.used)))

        if len(not_used) > 0:
            selected = not_used[0]
        else:
            shuffle(only_files)
            selected = only_files[0]

        if selected not in self.used:
            self.used.append(selected)
        with open("../log/datasets-used.txt", "w") as file:
            file.write(os.path.basename(__file__) + " " + selected)
        if downloader is not None:
            downloader.set_current_file(selected)
        return join(self.dir, selected)


def h5_small_vgg_generator(batch_size, dir, downloader):
    file_picker = H5Choose(dir=dir)
    x1 = None
    x2 = None
    y = None
    n = 0

    while True:
        if x1 is None or n > len(x1) - batch_size:
            file = file_picker.pick_next(downloader)
            x1 = HDF5Matrix(file, 'small')
            x2 = HDF5Matrix(file, 'vgg224')
            y = HDF5Matrix(file, 'ab_hist')
            n = 0
        yield [x1[n:n+batch_size], np.tile(x2[n:n+batch_size],  (1, 1, 1, 3))], y[n:n+batch_size]
        n += batch_size
