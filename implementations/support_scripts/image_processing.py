import os

import sys

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

import inspect
import threading

import numpy as np
import time

from os.path import isfile, join


from skimage import io, color

from random import shuffle, randint
import scipy.stats as st

import scipy.ndimage
import scipy.signal
import scipy.misc



from PIL import Image


# load data
from implementations.support_scripts.download_dataset import ImageDownloadGenerator


def load_images(dir, file, size=(256, 256)):
    try:
        img = Image.open(os.path.join(dir, file))
        img = img.resize(size, Image.ANTIALIAS)

    except (OSError, ValueError, IOError, ZeroDivisionError) as e:
        print(e)
        print("Damaged:", file)
        return "error"

    img = img.convert(mode="RGB")  # ensure that image rgb
    rgb = np.array(img)
    if len(rgb.shape) == 3 and (rgb.shape[2]) == 3:  # avoid black and white photos
        return color.rgb2lab(rgb)
    else:
        print(file)


def load_images_rgb(dir, file, size=(256, 256)):
    try:
        img = Image.open(os.path.join(dir, file))
        img = img.resize(size, Image.ANTIALIAS)

    except (OSError, ValueError, IOError):
        print("Damaged:", file)
        return "error"

    img = img.convert(mode="RGB")  # ensure that image rgb
    rgb = np.array(img)
    if len(rgb.shape) == 3 and (rgb.shape[2]) == 3:  # avoid black and white photos
        return rgb
    else:
        print(file)


def resize_image(im, size):
    return scipy.misc.imresize(im, size).astype(float) / 256 * 100
    # img = Image.fromarray(im, 'L')
    # img = img.resize(size, Image.ANTIALIAS)
    # return np.array(img)


def resize_rgb(im, size):
    img = Image.fromarray(im, 'RGB')
    img = img.resize(size, Image.ANTIALIAS)
    return np.array(img)


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


def ab_to_histogram_separate(ab):
    idxa = ((ab[:, :, 0] - a_range["min"]) / a_bin_len * n_bins).astype(int)
    idxb = ((ab[:, :, 1] - b_range["min"]) / b_bin_len * n_bins).astype(int)

    return np.concatenate((((np.arange(n_bins) == idxa[:, :, None]).astype(int)),
                          ((np.arange(n_bins) == idxb[:, :, None]).astype(int))), axis=2)


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


def histogram_to_ab_separate(hist):
    indicesa = np.argmax(hist[:, :, 0:20], axis=2)
    indicesb = np.argmax(hist[:, :, 20:40], axis=2)

    a = indicesa / n_bins * a_bin_len + a_range["min"]
    b = indicesb / n_bins * b_bin_len + b_range["min"]

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


def image_generator(image_dir, batch_size, im_size=(256, 256)):
    n = 0
    while True:
        batch_im_names = image_dir[n:n+batch_size]
        print(len(batch_im_names))

        batch_imputs = np.zeros((batch_size, im_size[0], im_size[0], 1))
        batch_targets = np.zeros((batch_size, im_size[0], im_size[0], 2))
        for i, image in enumerate(batch_im_names):
            im = load_images(image)
            batch_imputs[i] = images_to_l(im)[:, :, np.newaxis]
            batch_targets[i] = images_to_ab(im)

        yield (batch_imputs, batch_targets)
        n += batch_size
        if n + batch_size > len(image_dir):
            n = 0
            shuffle(image_dir)


def image_generator_parts(image_dir, batch_size, im_size=(256, 256), part_size=(32, 32)):
    n = 0
    while True:

        batch_imputs_part = np.zeros((batch_size, part_size[0], part_size[1], 1))
        batch_imputs_full = np.zeros((batch_size, im_size[0], im_size[1], 3))
        batch_targets = np.zeros((batch_size, part_size[0], part_size[1], 2))

        i = 0
        while i < batch_size:
            image = image_dir[n]
            n += 1
            if n >= len(image_dir):
                n = 0
                shuffle(image_dir)
            im = load_images("../small_dataset", image, size=im_size)
            if im == "error":
                continue
            random_i, random_j = randint(0, im.shape[0] - part_size[0]), randint(0, im.shape[1] - part_size[1])
            im_part = im[random_i : random_i + part_size[0], random_j : random_j + part_size[1], :]
            batch_imputs_part[i, :, :, :] = images_to_l(im_part)[:, :, np.newaxis]
            batch_targets[i, :, :, :] = images_to_ab(im_part)
            im_full = images_to_l(im)
            batch_imputs_full[i, :, :, :] = np.stack((im_full, im_full, im_full), axis=2)
            i += 1

        yield [batch_imputs_part, batch_imputs_full], batch_targets



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

    def __init__(self, dir, prefix, num_images=1024, num_files=None, mode="full-im", im_size=(256, 256)):
        super(ImageDownloader, self).__init__()
        self.things_lock = threading.Lock()
        self.dir = dir
        self.prefix = prefix
        self.n = self.find_n()
        self.image_generator = ImageDownloadGenerator()
        self.done = False
        self.num_images = num_images
        self.num_files = num_files
        self.current_file = ""
        self.mode = mode
        self.im_size = im_size

    def run(self):
        print('run')
        self.generate_files(self.num_images, self.num_files)

    def stop(self):
        print("stop")
        self.done = True

    def set_current_file(self, filename):
        self.current_file = filename

    def find_n(self):
        """
        This function finds n - number to start numbering the h5 files

        Returns
        -------
        int
            Number that tells from where to number the h5 files in dir
        """
        k = len(self.prefix)
        only_files = [f for f in os.listdir(self.dir) if isfile(join(self.dir, f)) and f[:k] == self.prefix]
        return max([-1] + [int(x[k:k + 4]) for x in only_files]) + 1  # files has name with format prefxxxx.h5 - x is a number

    def remove_oldest(self):
        k = len(self.prefix)
        only_files = sorted([f for f in os.listdir(self.dir) if isfile(join(self.dir, f)) and f[:k] == self.prefix])

        keep_files = 3 if self.mode == "separate" else (60 if self.mode == "small-vgg" else 10)

        while len(only_files) > keep_files and only_files[0] != self.current_file:
            os.remove(os.path.join(self.dir, only_files[0]))
            only_files = sorted([f for f in os.listdir(self.dir) if isfile(join(self.dir, f)) and f[:k] == self.prefix])

    def generate_files(self, num_images=1024, num_files=None):
        """
        Function generates h5 files with image data

        Parameters
        ----------
        num_images : int
            Number of images in one files
        num_files : int
            Number of files function generates before stop - if None continue until stopped
        """
        g = self.image_generator.download_images_generator()

        def gen():
            start = time.time()
            if self.mode == "separate":
                self.generate_h5_separate(g, num_images, "{}{:0=4d}.h5".format(self.prefix, self.n))
            elif self.mode == "small-vgg":
                self.generate_h5_small_vgg(g, num_images, "{}{:0=4d}.h5".format(self.prefix, self.n))
            else:
                self.generate_h5(g, num_images, "{}{:0=4d}.h5".format(self.prefix, self.n))
            self.n += 1
            print(time.time() - start)

        if num_files is None:
            while not self.done:
                gen()
                self.remove_oldest()
        else:
            for _ in range(num_files):
                if self.done:
                    break
                gen()
                self.remove_oldest()

    def generate_h5(self, generator, size, name):
        import h5py

        # generate examples
        x = np.zeros((size, self.im_size[0], self.im_size[0], 1))
        y = np.zeros((size, self.im_size[0], self.im_size[0]))
        print(y.shape)
        i = 0
        while i < size:
            # download image
            file_name = next(generator)
            script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
            rel_path = "./../small_dataset"
            abs_file_path = os.path.join(script_dir, rel_path)

            lab_im = load_images(abs_file_path, file_name, size=self.im_size)
            # remove already used file
            os.remove(os.path.join(abs_file_path, file_name))
            if lab_im == "error":
                continue

            x[i, :, :, :] = images_to_l(lab_im)[:, :, np.newaxis]
            y[i, :, :] = ab_to_indices(images_to_ab(lab_im))

            i += 1

        f = h5py.File(os.path.join(self.dir, name), 'w')
        # Creating dataset to store features
        X_dset = f.create_dataset('grayscale', (size, self.im_size[0], self.im_size[0], 1), dtype='float')
        X_dset[:] = x
        # Creating dataset to store labels
        y_dset = f.create_dataset('ab_hist', (size, self.im_size[0], self.im_size[0]), dtype='int32')
        y_dset[:] = y
        f.close()

    def generate_h5_small_vgg(self, generator, size, name):
        import h5py

        # generate examples
        x1 = np.zeros((size, 32, 32, 1))
        x2 = np.zeros((size, 224, 224, 1))
        y = np.zeros((size, 32, 32, 2))

        i = 0
        while i < size:
            # download image
            file_name = next(generator)
            script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
            rel_path = "./../small_dataset"
            abs_file_path = os.path.join(script_dir, rel_path)

            lab_im = load_images(abs_file_path, file_name, size=self.im_size)

            # remove already used file
            # os.remove(os.path.join(abs_file_path, file_name))  # commented too keep files
            if lab_im == "error":
                continue

            h, w, _ = lab_im.shape

            random_i, random_j = randint(0, h - 32), randint(0, w - 32)
            im_part = lab_im[random_i: random_i + 32, random_j: random_j + 32, :]

            x1[i, :, :, :] = images_to_l(im_part)[:, :, np.newaxis]
            x2[i, :, :, :] = images_to_l(lab_im)[:, :, np.newaxis]
            y[i, :, :] = images_to_ab(im_part)

            i += 1

        # print(np.mean(y ** 2))

        f = h5py.File(os.path.join(self.dir, name), 'w')
        # Creating dataset to store features
        X1_dset = f.create_dataset('small', (size, 32, 32, 1), dtype='float')
        X1_dset[:] = x1

        X2_dset = f.create_dataset('vgg224', (size, self.im_size[0], self.im_size[0], 1), dtype='float')
        X2_dset[:] = x2
        # Creating dataset to store labels
        y_dset = f.create_dataset('ab_hist', (size, 32, 32, 2), dtype='float')
        y_dset[:] = y
        f.close()

    def generate_h5_separate(self, generator, size, name):
        import h5py

        # generate examples
        x = np.zeros((size, 256, 256, 1))
        y = np.zeros((size, 256, 256, 40), dtype=bool)
        print("new download")
        for i in range(size):
            # download image
            file_name = next(generator)

            script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
            rel_path = "./../small_dataset"
            abs_file_path = os.path.join(script_dir, rel_path)

            lab_im = load_images(abs_file_path, file_name)
            x[i, :, :, :] = images_to_l(lab_im)[:, :, np.newaxis]
            y[i, :, :, :] = ab_to_histogram_separate(images_to_ab(lab_im)).astype(bool)


        f = h5py.File(os.path.join(self.dir, name), 'w')
        # Creating dataset to store features
        X_dset = f.create_dataset('grayscale', (size, 256, 256, 1), dtype='float')
        X_dset[:] = x
        # Creating dataset to store labels
        y_dset = f.create_dataset('ab_hist', (size, 256, 256, 40), dtype='u1')
        y_dset[:] = y
        f.close()

if __name__ == "__main__":
    id = ImageDownloader("../../h5_data", "imp7d-", num_images=1024, num_files=10, im_size=(224, 224), mode="small-vgg")
    id.run() # todo: debug that


