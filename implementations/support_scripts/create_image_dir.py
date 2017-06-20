import os

import pickle
from random import choice

import numpy as np
from shutil import copyfile

dataset_dir = "../../../imagenet"
dir_to = "../../../subset100_000"
train_set_len = 100000
validation_set_len = 10000

""" This part list the dir and count number of images in subdirectories """

# image_dirs = os.listdir(dataset_dir)
#
# sizes = []
# for i, im_dir in enumerate(image_dirs):
#     sizes.append(len(os.listdir(os.path.join(dataset_dir, im_dir))))
#     if i % 100 == 0:
#         print(i)
#
# if not os.path.isfile(dir_to):
#     os.mkdir(dir_to)
file_name = os.path.join(dir_to, "file_dist.pkl")

# save to pickle that new calculation is not required every time
# with open(file_name, 'wb') as handle:
#     pickle.dump((image_dirs, sizes), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(file_name, 'rb') as handle:
    image_dirs, sizes = pickle.load(handle)

# just test if everything is right with sizes
print(sum(sizes))

""" Script to generate a subset """

# calculate probabilities for a directory, more files higher probability for image to be chosen
sizes = np.array(sizes)
dir_probabilities = sizes / np.sum(sizes)

# check probabilities if sum in 1
assert(abs(sum(dir_probabilities) - 1) < 0.0001)

# get random selection of directories
dir_choices_train = np.random.choice(image_dirs, train_set_len, p=dir_probabilities)
dir_choices_validation = np.random.choice(image_dirs, train_set_len, p=dir_probabilities)

# create directories to save image if does not exist
if not os.path.isfile(os.path.join(dir_to, "train")):
    os.mkdir(os.path.join(dir_to, "train"))
if not os.path.isfile(os.path.join(dir_to, "validation")):
    os.mkdir(os.path.join(dir_to, "validation"))

# copy files to dir and write file with names
with open(os.path.join(dir_to, "train.txt"), 'w') as handle:
    for i, d in enumerate(dir_choices_train):
        images = os.listdir(os.path.join(dataset_dir, d))
        ch = choice(images)
        copyfile(os.path.join(dataset_dir, d, ch), os.path.join(dir_to, "train", ch))
        print(ch, file=handle)
        if i % 1000 == 0:
            print(i)

with open(os.path.join(dir_to, "train.txt"), 'w') as handle:
    for i, d in enumerate(dir_choices_validation):
        images = os.listdir(os.path.join(dataset_dir, d))
        ch = choice(images)
        copyfile(os.path.join(dataset_dir, d, ch), os.path.join(dir_to, "validation", ch))
        print(ch, file=handle)
        if i % 1000 == 0:
            print(i)
