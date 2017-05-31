import numpy as np
from skimage import io, color
from skimage.transform import resize
from random import shuffle
import os
import scipy.stats as st

import scipy.ndimage
# import matplotlib.pyplot as plt
# from scipy.misc import imread


from PIL import Image


# load data
def load_images(dir, file):
    images = []

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



def ab_to_histogram(ab):

    indices = ((ab[:, :, 0] - a_range["min"]) / a_bin_len * n_bins).astype(int) * n_bins + \
              ((ab[:, :, 1] - b_range["min"]) / b_bin_len * n_bins).astype(int)

    return ((np.arange(n_bins ** 2) == indices[:, :, None]).astype(int))


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
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            im = np.reshape(data[i, j, :], (n_bins, n_bins))
            k = gkern(3, nsig=sigma)
            l = scipy.ndimage.convolve(im.astype(float), k, mode='constant', cval=0.0)
            data[i, j, :] = l.flatten()
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
            batch_targets[i] = ab_to_histogram(images_to_ab(im))
            # in case user want smoothed targets
            if mode == "gaussian":
                batch_targets[i] = gaussian_filter(batch_targets[i])
        yield (batch_imputs, batch_targets)
        n += batch_size
        if n + batch_size > len(image_dir):
            n = 0
            shuffle(image_dir)


if __name__ == "__main__":
    # test
    a = np.array([[[0, 0], [-87, -108], [98, 94]],
                  [[5, 10], [11, 8], [-10, -23]]])
    h = ab_to_histogram(a)
    c = histogram_to_ab(h)
    print(c)




