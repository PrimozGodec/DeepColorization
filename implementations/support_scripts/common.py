from random import shuffle

import numpy as np
import scipy.misc
from os import listdir

from os.path import isfile, join

import time
from skimage import color
import keras.backend as K

"""
This script contains functions that are common to all the implementations used
"""
from implementations.support_scripts.image_processing import image_generator_hist, histogram_to_ab, \
    histogram_to_ab_separate


def make_prediction_sample(model, batch_size, name):
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


def data_to_onehot(data):
    t = K.one_hot(K.round(data), 400)
    tf_session = K.get_session()
    return t.eval(session=tf_session)

class H5Choose:

    def __init__(self, dir):
        self.dir = dir
        self.used = []

    def all_files(self):
        return [f for f in listdir(self.dir) if isfile(join(self.dir, f)) and f.endswith("h5")]

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
        print("Selected dataset: ", selected)
        downloader.set_current_file(selected)
        return join(self.dir, selected)
