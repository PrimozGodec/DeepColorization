import inspect
import os
import random

import h5py
import numpy as np
import scipy.misc
from skimage import color


class VideoH5Chooser:

    def __init__(self, dir, random):
        self.dir = dir
        self.used = []
        self.random = random
        self.n = 0
        self.files = self.all_files()

    def all_files(self):
        return [f for f in os.listdir(self.dir) if f.endswith("h5")]

    def pick_next(self):
        if not self.random:
            sel = os.path.join(self.dir, self.files[self.n])
            self.n += 1
            if self.n >= len(self.files):
                self.n = 0
        else:
            to_select = list(set(self.files) - set(self.used))

            sel = random.choice(to_select)
            self.used.append(sel)
            if len(to_select) <= 1:
                self.used = []  # to start again

        return os.path.join(self.dir, sel)


def video_imp9_full_generator(batch_size, dir, num_neighbours=0, random=True):
    file_picker = VideoH5Chooser(dir=dir, random=random)
    x1 = None
    n = 0
    f = None

    while True:
        if x1 is None or n > len(x1) - batch_size - num_neighbours:
            if f is not None:
                f.close()
            file = file_picker.pick_next()
            f = h5py.File(file, 'r')
            x1 = f['im']
            n = num_neighbours  # skip fist n images if learning with neighbours

        frames = []
        for i in range(batch_size):
            frames.append(np.stack(
                np.split(x1[n+i-num_neighbours:n+i+num_neighbours+1, :, :, 0], 1+2*num_neighbours, axis=0), axis=3))

        frames = np.stack(frames, axis=0)
        b, _, w, h, c = frames.shape

        yield ([frames.reshape(b, w, h, c),
            np.tile(x1[n:n+batch_size, :, :, 0][:, :, :, np.newaxis],  (1, 1, 1, 3))],
            x1[n:n+batch_size, :, :, 1:3])
        n += batch_size


def video_visual_checker_imp9_full(model, num_of_images, name, b_size=32, num_neighbours=0):
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
    g = video_imp9_full_generator(num_of_images, "../data/video/training", num_neighbours=num_neighbours)

    network_input = next(g)[0]

    color_im = model.predict(network_input, batch_size=b_size)

    for i in range(num_of_images):
        # to rgb
        lab_im = np.concatenate((network_input[0][i, :, :, 0][:, :, np.newaxis], color_im[i]), axis=2)
        im_rgb = color.lab2rgb(lab_im)

        # save
        abs_svave_path = os.path.join(script_dir, '../../result_images/')
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + name + str(i) + ".jpg")


# test
if __name__ == "__main__":
    # v = VideoH5Chooser("../../h5_data", False)
    # for i in range(10):
    #     print(v.pick_next())

    g = h5_vgg_generator_let_there(2, "../../data/video/training", num_neighbours=1)
    for i in range(2):
        a = next(g)

    print(a[0][0].shape)

    # test
    f = h5py.File("../../data/video/training/0.h5", 'r')
    x1 = f['im']

    print(np.sum(a[0][0][0, :, :, 1] - x1[3, :, :, 0]))

