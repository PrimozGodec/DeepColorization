"""
This script resizes all videos to size 224 x 244
Transform frames to lab
Write them as h5 file
"""
import os
import sys

import h5py
import numpy as np
import skvideo.io
from skimage import color

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])
from implementations.support_scripts.image_processing import resize_rgb


def save_h5(data, to_dir, im_num):
    f = h5py.File(os.path.join(to_dir, str(im_num) + ".h5"), 'w')
    f.create_dataset('im', data.shape, dtype='float', data=data)


def video2h5(from_dir, to_dir, images_per_file):

    # list files to pack
    video_files = os.listdir(from_dir)

    # create empty list for rames
    frames_lab = np.zeros((images_per_file, 224, 224, 3))
    file_n = 0
    n = 0

    for i, video_file in enumerate(video_files):
        vid = skvideo.io.vreader(os.path.join(from_dir, video_file))
        for frame in vid:
            rgb = resize_rgb(frame, size=(224, 224))
            lab = color.rgb2lab(rgb)

            frames_lab[n, :, :, :] = lab
            n += 1

            if n >= images_per_file:
                save_h5(frames_lab, to_dir, file_n)
                file_n += 1
                n = 0

        vid.close()
        print(i)

    # save last data if exist
    if n >= 0:
        save_h5(frames_lab[:n, :, :, :], to_dir, file_n)


if __name__ == "__main__":
    video2h5("../../../videos_dataset/validation", "../../../videos_dataset/validation_h5", 1024)