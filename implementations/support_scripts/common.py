from random import shuffle

import h5py
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
    histogram_to_ab_separate, load_images, images_to_l, resize_image, lab2rgb


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


def image_check(model, num_of_images, name):
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
    rel_path = "../../test_set"
    abs_file_path = os.path.join(script_dir, rel_path)
    image_list = os.listdir(abs_file_path)

    all_images = np.zeros((num_of_images, 224, 224, 3))
    for i in range(num_of_images):
        # get image
        image_lab = load_images(abs_file_path, image_list[i], size=(224, 224))  # image is of size 256x256
        image_l = images_to_l(image_lab)
        all_images[i, :, :, :] = np.tile(image_l[:, :, np.newaxis], (1, 1, 1, 3))

    color_im = model.predict(all_images)

    for i in range(num_of_images):
        # to rgb
        print(all_images[i, :, :, 0][:, :, np.newaxis])
        color_im = np.concatenate((all_images[i, :, :, 0][:, :, np.newaxis], color_im[i, :, :, :]), axis=2)
        im_rgb = color.lab2rgb(color_im)

        # save
        abs_svave_path = os.path.join(script_dir, '../../result_images/')
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + name + image_list[i])


def whole_image_check(model, num_of_images, name):

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


# matrices for multiplying that needs to calculate only once
vec = np.hstack((np.linspace(1/16, 1 - 1/16, 16), np.flip(np.linspace(1/16, 1 - 1/16, 16), axis=0)))
one = np.ones((32, 32))
xv, yv = np.meshgrid(vec, vec)
weight_m = xv * yv
weight_left = np.hstack((one[:, :16], xv[:, 16:])) * yv
weight_right = np.hstack((xv[:, :16], one[:, 16:])) * yv
weight_top = np.vstack((one[:16, :], yv[16:, :])) * xv
weight_bottom = np.vstack((yv[:16, :], one[16:, :])) * xv

weight_top_left = np.hstack((one[:, :16], xv[:, 16:])) * np.vstack((one[:16, :], yv[16:, :]))
weight_top_right = np.hstack((xv[:, :16], one[:, 16:])) * np.vstack((one[:16, :], yv[16:, :]))
weight_bottom_left = np.hstack((one[:, :16], xv[:, 16:])) * np.vstack((yv[:16, :], one[16:, :]))
weight_bottom_right = np.hstack((xv[:, :16], one[:, 16:])) * np.vstack((yv[:16, :], one[16:, :]))


def whole_image_check_overlapping(model, num_of_images, name):

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
        slices = np.zeros((slices_dim * slices_dim * 4, 32, 32, 1))
        for a in range(slices_dim * 2 - 1):
            for b in range(slices_dim * 2 - 1):

                slices[a * slices_dim * 2 + b] = image_l[a*32//2: a*32//2 + 32, b*32//2: b*32//2 + 32, np.newaxis]

        # lover originals dimension to 224x224 to feed vgg and increase dim
        image_l_224_b = resize_image(image_l, (224, 224))
        image_l_224 = np.repeat(image_l_224_b[:, :, np.newaxis], 3, axis=2).astype(float)


        # append together booth lists
        input_data = [slices, np.array([image_l_224,] * slices_dim ** 2 * 4)]

        # predict
        predictions_ab = model.predict(input_data, batch_size=32)

        # reshape back
        original_size_im = np.zeros((h, w, 2))

        for n in range(predictions_ab.shape[0]):
            a, b = n // (slices_dim * 2) * 16, n % (slices_dim * 2) * 16

            if a + 32 > 256 or b + 32 > 256:
                continue  # it is empty edge

            # weight decision
            if a == 0 and b == 0:
                weight = weight_top_left
            elif a == 0 and b == 224:
                weight = weight_top_right
            elif a == 0:
                weight = weight_top
            elif a == 224 and b == 0:
                weight = weight_bottom_left
            elif b == 0:
                weight = weight_left
            elif a == 224 and b == 224:
                weight = weight_bottom_right
            elif a == 224:
                weight = weight_bottom
            elif b == 224:
                weight = weight_right
            else:
                weight = weight_m

            im_a = predictions_ab[n, :, :, 0] * weight
            im_b = predictions_ab[n, :, :, 1] * weight

            original_size_im[a:a+32, b:b+32, :] += np.stack((im_a, im_b), axis=2)

        # to rgb
        color_im = np.concatenate((image_l[:, :, np.newaxis], original_size_im), axis=2)
        # color_im = np.concatenate(((np.ones(image_l.shape) * 50)[:, :, np.newaxis], original_size_im), axis=2)
        im_rgb = color.lab2rgb(color_im)

        # save
        abs_svave_path = os.path.join(script_dir, '../../result_images/')
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + name + image_list[i])


def whole_image_check_hist(model, num_of_images, name):

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
                slices[a * slices_dim + b] = image_l[a * 32: (a + 1) * 32, b * 32 : (b + 1) * 32, np.newaxis]

        # lover originals dimension to 224x224 to feed vgg and increase dim
        image_l_224_b = resize_image(image_l, (224, 224))
        image_l_224 = np.repeat(image_l_224_b[:, :, np.newaxis], 3, axis=2).astype(float)


        # append together booth lists
        input_data = [slices, np.array([image_l_224,] * slices_dim ** 2)]

        # predict
        predictions_hist = model.predict(input_data)

        # reshape back
        # predictions_a = np.argmax(predictions_hist[:, :, :, :20], axis=3) * 10 - 100 + 5
        # predictions_b = np.argmax(predictions_hist[:, :, :, 20:], axis=3) * 10 - 100 + 5  # +5 to set in the middle box
        indices = np.argmax(predictions_hist[:, :, :, :], axis=3)

        predictions_a = indices // 20 * 10 - 100 + 5
        predictions_b = indices % 20 * 10 - 100 + 5  # +5 to set in the middle box

        predictions_ab = np.stack((predictions_a, predictions_b), axis=3)
        original_size_im = np.zeros((h, w, 2))
        for n in range(predictions_ab.shape[0]):
            a, b = n // slices_dim * 32, n % slices_dim * 32
            original_size_im[a:a+32, b:b+32, :] = predictions_ab[n, :, :, :]

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
        # with open("../log/datasets-used.txt", "w") as file:
        #     file.write(os.path.basename(__file__) + " " + selected)
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
        f = None
        if x1 is None or n > len(x1) - batch_size:
            if f is not None:
                f.close()
            file = file_picker.pick_next(downloader)
            f = h5py.File(file, 'r')
            x1 = f['small']
            x2 = f['vgg224']
            y = f['ab_hist']
            n = 0
        yield [x1[n:n+batch_size], np.tile(x2[n:n+batch_size],  (1, 1, 1, 3))], y[n:n+batch_size]
        n += batch_size


def h5_vgg_generator(batch_size, dir, downloader):
    file_picker = H5Choose(dir=dir)
    x1 = None
    n = 0
    f = None

    while True:
        if x1 is None or n > len(x1) - batch_size:
            if f is not None:
                f.close()
            file = file_picker.pick_next(downloader)
            f = h5py.File(file, 'r')
            x1 = f['im']
            n = 0

        yield np.tile(x1[n:n+batch_size, :, :, 0][:, :, :, np.newaxis],  (1, 1, 1, 3)), x1[n:n+batch_size, :, :, 1:3]
        n += batch_size


def h5_small_vgg_generator_onehot(batch_size, dir, downloader):

    # def to_one_hot(x):
    #     def one_hot(indices, num_classes):
    #         return (np.arange(num_classes) == indices[:, :, :, None]).astype(int)
    #
    #     a = one_hot(((x[:, :, :, 0] + 100) / 10).astype(int), 20)  # 20 bins
    #     b = one_hot(((x[:, :, :, 1] + 100) / 10).astype(int), 20)
    #     return np.concatenate((a, b), axis=3)

    def to_one_hot(x):
        def one_hot(indices, num_classes):
            return (np.arange(num_classes) == indices[:, :, :, None]).astype(int)

        a = one_hot(((x[:, :, :, 0] + 100) / 10).astype(int) * 20 +
                    ((x[:, :, :, 1] + 100) / 10).astype(int), 400)  # 20 bins
        return a


    file_picker = H5Choose(dir=dir)
    x1 = None
    x2 = None
    y = None
    n = 0
    f = None

    while True:
        if x1 is None or n > len(x1) - batch_size:
            if f is not None:
                f.close()
            file = file_picker.pick_next(downloader)
            f = h5py.File(file, 'r')
            x1 = f['small']
            x2 = f['vgg224']
            y = f['ab_hist']
            n = 0
        yield [x1[n:n+batch_size], np.tile(x2[n:n+batch_size],  (1, 1, 1, 3))], to_one_hot(y[n:n+batch_size])
        n += batch_size



if __name__ == "__main__":
    g = h5_small_vgg_generator(2, "../../h5_data", None)
    g1 = h5_vgg_generator(2, "../../h5_data_224", None)
    import matplotlib.pyplot as plt
    for i in range(50):
        a = next(g)
        print(a[0][1].shape)

        a = next(g1)
        print(a[0].shape)
        exit()