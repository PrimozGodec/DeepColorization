import inspect
import os
import random
import threading

from os.path import isfile, join
import numpy as np
import time

import sys

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])
from implementations.support_scripts.image_processing import load_images, images_to_l, images_to_ab


class ImagePacker(threading.Thread):
    """
    This class is used to pack images to h5 files
    """

    def __init__(self, dir_from, dir_to, prefix, num_images=1024, num_files=None):
        super(ImagePacker, self).__init__()
        self.dir_from = dir_from
        self.dir_to = dir_to
        self.prefix = prefix
        self.n = self.find_n()
        self.done = False
        self.num_images = num_images
        self.num_files = num_files
        self.current_file = ""
        self.images_list = []
        self.im_size = (224, 224)

    def run(self):
        print('run')
        self.generate_files()

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
        only_files = [f for f in os.listdir(self.dir_to) if isfile(join(self.dir_to, f)) and f[:k] == self.prefix]
        return max([-1] + [int(x[k:k + 4]) for x in only_files]) + 1  # files has name with format prefxxxx.h5 - x is a number

    def remove_oldest(self):
        """
        Function removes files that are the oldest to release some space
        """
        def get_files():
            return sorted([f for f in os.listdir(self.dir_to) if isfile(join(self.dir_to, f)) and f[:k] == self.prefix])

        k = len(self.prefix)
        only_files = get_files()

        keep_files = 2000

        while len(only_files) > keep_files and only_files[0] != self.current_file:
            os.remove(os.path.join(self.dir_to, only_files[0]))
            only_files = get_files()

    def generate_files(self):
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
            start = time.time()
            self.generate_h5_small_vgg(self.num_images, "{}{:0=4d}.h5".format(self.prefix, self.n))

            self.n += 1
            print("New file", time.time() - start)

        # load list of files only once
        self.images_list = os.listdir(self.dir_from)

        if self.num_files is None:
            while not self.done:
                gen()
                self.remove_oldest()
        else:
            for _ in range(self.num_files):
                if self.done:
                    break
                gen()
                self.remove_oldest()

    def select_file(self):
        return random.choice(self.images_list)

    def generate_h5_small_vgg(self, size, name):
        import h5py

        # generate examples
        x1 = np.zeros((size, 32, 32, 1))
        x2 = np.zeros((size, 224, 224, 1))
        y = np.zeros((size, 32, 32, 2))

        i = 0
        while i < size:
            # print(i)
            # download image
            file_name = self.select_file()
            lab_im = load_images(self.dir_from, file_name, size=self.im_size)

            if type(lab_im) is not np.ndarray or lab_im == "error":
                continue

            h, w, _ = lab_im.shape

            random_i, random_j = random.randint(0, h - 32), random.randint(0, w - 32)
            im_part = lab_im[random_i: random_i + 32, random_j: random_j + 32, :]

            x1[i, :, :, :] = images_to_l(im_part)[:, :, np.newaxis]
            x2[i, :, :, :] = images_to_l(lab_im)[:, :, np.newaxis]
            y[i, :, :] = images_to_ab(im_part)

            i += 1

        f = h5py.File(os.path.join(self.dir_to, name), 'w')
        # Creating dataset to store features
        X1_dset = f.create_dataset('small', (size, 32, 32, 1), dtype='float')
        X1_dset[:] = x1

        X2_dset = f.create_dataset('vgg224', (size, self.im_size[0], self.im_size[0], 1), dtype='float')
        X2_dset[:] = x2
        # Creating dataset to store labels
        y_dset = f.create_dataset('ab_hist', (size, 32, 32, 2), dtype='float')
        y_dset[:] = y
        f.close()


if __name__ == "__main__":
    ip = ImagePacker("../../small_dataset", "../../h5_data",  "imp7d-big-", num_images=10240, num_files=None)
    ip.start()