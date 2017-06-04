import inspect
import threading

import numpy as np
import time

from os.path import isfile, join
from skimage import io, color
from skimage.transform import resize
from random import shuffle
import os
import scipy.stats as st

import scipy.ndimage
import scipy.signal
# import matplotlib.pyplot as plt
# from scipy.misc import imread


from PIL import Image


# load data
from implementations.support_scripts.download_dataset import ImageDownloadGenerator


def load_images(dir, file):

    rgb = io.imread(os.path.join(dir, file))
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


def ab_to_indices(ab, mode="one-hot"):

    return ((ab[:, :, 0] - a_range["min"]) / a_bin_len * n_bins).astype(int) * n_bins + \
              ((ab[:, :, 1] - b_range["min"]) / b_bin_len * n_bins).astype(int)


def ab_to_histogram(ab, mode="one-hot"):

    indices = ((ab[:, :, 0] - a_range["min"]) / a_bin_len * n_bins).astype(int) * n_bins + \
              ((ab[:, :, 1] - b_range["min"]) / b_bin_len * n_bins).astype(int)

    h, w, d = ab.shape
    im = np.zeros((h, w, n_bins ** 2))


    if mode == "gaussian":
        kernel = gkern(3, nsig=5)
        top_left = ((np.arange(n_bins ** 2) == indices[:, :, None] - n_bins - 1).astype(float)) * kernel[0, 0]
        top = ((np.arange(n_bins ** 2) == indices[:, :, None] - n_bins).astype(float)) * kernel[0, 1]
        top_right = ((np.arange(n_bins ** 2) == indices[:, :, None] - n_bins + 1).astype(float)) * kernel[0, 2]
        right = ((np.arange(n_bins ** 2) == indices[:, :, None] + 1).astype(float)) * kernel[1, 2]
        left = ((np.arange(n_bins ** 2) == indices[:, :, None] - 1).astype(float)) * kernel[1, 0]
        bottom_left = ((np.arange(n_bins ** 2) == indices[:, :, None] + n_bins - 1).astype(float)) * kernel[2, 0]
        bottom = ((np.arange(n_bins ** 2) == indices[:, :, None] + n_bins).astype(float)) * kernel[2, 1]
        bottom_right = ((np.arange(n_bins ** 2) == indices[:, :, None] + n_bins + 1).astype(float)) * kernel[2, 2]
        im += top_left + top + top_right + left + right + bottom_left + bottom + bottom_right

    return ((np.arange(n_bins ** 2) == indices[:, :, None]).astype(int)) + im



def histogram_to_ab(hist):
    indices = np.argmax(hist, axis=2)

    a = (indices // n_bins) / n_bins * a_bin_len + a_range["min"]
    b = (indices % n_bins) / n_bins * b_bin_len + b_range["min"]

    return np.stack((a, b), axis=2)


def gkern(kernlen, nsig=5):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def gaussian_filter(data, sigma=5):
    data = data.astype(np.float32)
    w, h, c = data.shape
    data = data.reshape((w * h, int(np.sqrt(c)), int(np.sqrt(c))))

    k = gkern(3, nsig=sigma)[None, :, :]
    l = scipy.signal.convolve(data.astype(float), k, mode='same')

    return l.reshape(w, h, c)


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


def image_generator_hist(image_dir, image_dir_name, batch_size, mode="one-hot"):
    """
    Return batches of images

    Parameters
    ----------
    image_dir
    batch_size
    mode : string
        String defining if ground truth is one-hot vector or filtered with gaussian filter
        Options: one-hot, gaussian

    Returns
    -------

    """
    if image_dir is None:
        image_dir = os.listdir(image_dir_name)
        shuffle(image_dir)
    n = 0
    while True:
        batch_im_names = image_dir[n:n+batch_size]
        batch_imputs = np.zeros((batch_size, 256, 256, 1))
        batch_targets = np.zeros((batch_size, 256, 256, 400))
        for i, image in enumerate(batch_im_names):
            im = load_images(image_dir_name, image)
            batch_imputs[i] = images_to_l(im)[:, :, np.newaxis]
            batch_targets[i] = ab_to_histogram(images_to_ab(im), mode=mode)

        yield (batch_imputs, batch_targets)
        n += batch_size
        if n + batch_size > len(image_dir):
            n = 0
            shuffle(image_dir)


class ImageDownloader(threading.Thread):
    """
    This class is used to download images and save them in H5 files
    """

    def __init__(self, dir, num_images=1024, num_files=None):
        super(ImageDownloader, self).__init__()
        self.things_lock = threading.Lock()
        self.dir = dir
        self.n = self.find_n()
        self.image_generator = ImageDownloadGenerator()
        self.done = False
        self.num_images = num_images
        self.num_files = num_files

    def run(self):
        print('run')
        self.generate_files(self.num_images, self.num_files)

    def stop(self):
        print("stop")
        self.done = True

    def find_n(self):
        """
        This function finds n - number to start numbering the h5 files

        Returns
        -------
        int
            Number that tells from where to number the h5 files in dir
        """

        only_files = [f for f in os.listdir(self.dir) if isfile(join(self.dir, f))]
        return max([int(x[:4]) for x in only_files])  # files has name with format xxxx.h5 - x is a number

    def generate_files(self, num_images = 1024, num_files=None):
        """
        Function generates h5 files with image data

        Parameters
        ----------
        num_images : int
            Number of images in one files
        num_files : int
            Number of files function generates before stop - if None continue until stopped
        """

        def gen():
            g = self.image_generator.download_images_generator()
            self.generate_h5(g, num_images, "{0:0=4d}.h5".format(self.n))
            self.n += 1

        if num_files is None:
            while not self.done:
                gen()
        else:
            for _ in range(num_files):
                if self.done:
                    break
                gen()


    def generate_h5(self, generator, size, name):
        import h5py

        # generate examples
        x = np.zeros((size, 256, 256, 1))
        y = np.zeros((size, 256, 256))

        for i in range(size):
            # download image
            file_name = next(generator)

            script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
            rel_path = "./../small_dataset"
            abs_file_path = os.path.join(script_dir, rel_path)

            lab_im = load_images(abs_file_path, file_name)
            x[i, :, :, :] = images_to_l(lab_im)[:, :, np.newaxis]
            y[i, :, :] = ab_to_indices(images_to_ab(lab_im))

        f = h5py.File(os.path.join(self.dir, name), 'w')
        # Creating dataset to store features
        X_dset = f.create_dataset('grayscale', (size, 256, 256, 1), dtype='float')
        X_dset[:] = x
        # Creating dataset to store labels
        y_dset = f.create_dataset('ab_hist', (size, 256, 256), dtype='int32')
        y_dset[:] = y
        f.close()



if __name__ == "__main__":
    id = ImageDownloader("../../h5_data", num_images=2)
    id.start() # todo: debug that

