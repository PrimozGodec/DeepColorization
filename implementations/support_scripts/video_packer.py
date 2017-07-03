"""
This script resizes all videos to size 224 x 244
Transform frames to lab
Write them as h5 file
"""
import os
import sys
from random import randint

import h5py
import numpy as np

import time


sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

num_neighbours = 4

def save_h5(data, to_dir, im_num):
    l, ab = data
    f = h5py.File(os.path.join(to_dir, str(im_num) + ".h5"), 'w')
    f.create_dataset('l', l.shape, dtype='float', data=l)
    f.create_dataset('ab', ab.shape, dtype='float', data=ab)
    f.close()


def video2h5(from_dir, to_dir, images_per_file, num_files):

    # list files to pack
    video_files = os.listdir(from_dir)

    # for each h5 file
    for file_n in range(num_files):
        t = time.time()
        frames_l = np.zeros((images_per_file, 224, 224, 9))
        frames_ab = np.zeros((images_per_file, 224, 224, 2))

        # make random choice of video files to be packed
        selected_videos = np.random.choice(video_files, 1024)
        for i, file_from in enumerate(selected_videos.tolist()):
            file_path = os.path.join(from_dir, file_from)
            f = h5py.File(file_path, 'r')
            images = f["im"]
            T = images.shape[0]

            r_frame = randint(0, T-1-(num_neighbours * 2 + 1))
            lab_vid = images[r_frame:r_frame+(num_neighbours*2 + 1)]
            # print(lab_vid.shape)
            # save middle image as ab
            frames_ab[i, :, :, :] = lab_vid[num_neighbours, :, :, 1:]
            stacked_l = np.stack(np.split(lab_vid[:, :, :, 0], 1 + 2 * num_neighbours, axis=0), axis=3)
            frames_l[i, :, :, :] = stacked_l


        save_h5((frames_l, frames_ab), to_dir, file_n)
        print(time.time() - t)


if __name__ == "__main__":
    video2h5("../../../videos_dataset/training_h5", "../../data/video/training", 1024, 100)