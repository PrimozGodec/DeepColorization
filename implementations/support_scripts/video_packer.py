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
import skvideo.io
from skimage import color
import time
import pickle


sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])
from implementations.support_scripts.image_processing import resize_rgb

num_neighbours = 4

def save_h5(data, to_dir, im_num):
    l, ab = data
    f = h5py.File(os.path.join(to_dir, str(im_num) + ".h5"), 'w')
    f.create_dataset('l', l.shape, dtype='float', data=l)
    f.create_dataset('ab', ab.shape, dtype='float', data=ab)


def video2h5(from_dir, to_dir, images_per_file, num_files, name):

    # list files to pack
    video_files = os.listdir(from_dir)

    # make file with video lengths
    file_save_lengths = "../../processed_data/video_frame_num_%s.pkl" % name
    if not os.path.isfile(file_save_lengths):
        # obtain num frames for each video
        num_fr = dict()
        for v in video_files:
            r = skvideo.io.FFmpegReader(os.path.join(from_dir, v))
            num_fr[v] = r.getShape()[0]
            print(v)
        with open(file_save_lengths, "wb") as f:
            pickle.dump(num_fr, f)

    with open(file_save_lengths, "rb") as f:
        num_fr = pickle.load(f)

    # create empty list for frames

    n = 0

    # for each h5 file
    for file_n in range(num_files):
        frames_l = np.zeros((images_per_file, 224, 224, 9))
        frames_ab = np.zeros((images_per_file, 224, 224, 2))

        # make random choice of video files to be packed
        selected_videos = np.random.choice(video_files, 1024)
        for i, video in enumerate(selected_videos.tolist()):
            t = time.time()
            file_path = os.path.join(from_dir, video)

            T = num_fr[video]
            r_frame = randint(0, T-1-(num_neighbours * 2 - 1))
            vid = skvideo.io.vread(file_path, outputdict={"-vf": "select=gte(n\,%d)" % r_frame}, num_frames=num_neighbours*2+1)
            print(vid.shape)
            print("a", time.time() - t)
            # resize images
            lab_vid = np.zeros((vid.shape[0], 224, 224, 3))
            for frame_num in range(vid.shape[0]):
                rgb = resize_rgb(vid[frame_num], size=(224, 224))
                lab_vid[frame_num, :, :, :] = color.rgb2lab(rgb)
            print("b", time.time() - t)
            # save middle image as ab
            frames_ab[i, :, :, :] = lab_vid[num_neighbours, :, :, 1:]
            stacked_l = np.stack(np.split(lab_vid[:, :, :, 0], 1 + 2 * num_neighbours, axis=0), axis=3)
            frames_l[i, :, :, :] = stacked_l
            print("c", time.time() - t)

        save_h5((frames_l, frames_ab), to_dir, file_n)


if __name__ == "__main__":
    video2h5("../../../videos_dataset/training", "../../data/video/training", 1024, 100, "training")