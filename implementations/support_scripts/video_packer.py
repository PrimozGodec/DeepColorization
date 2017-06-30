"""
This script resizes all videos to size 224 x 244
Transform frames to lab
Write them as h5 file
"""
import os
import numpy as np
import cv2
import skvideo.io


def video2h5(from_dir, to_dir, images_per_file):

    # list files to pack
    video_files = os.listdir(from_dir)

    # create empty list for rames
    frames_lab = np.zeros((images_per_file, 224, 224, 3))
    n = 0

    for video_file in video_files:
        vid = skvideo.io.vreader(os.path.join(from_dir, video_file))
        for frame in vid:
            print(frame.shape)
        vid.close()
        exit()




if __name__ == "__main__":
    video2h5("../../../videos_dataset/training", "../../data/video/training", 1024)