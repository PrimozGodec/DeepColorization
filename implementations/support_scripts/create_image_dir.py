import os

import pickle
from random import choice

import numpy as np
from shutil import copyfile

import time
from PIL import Image

dataset_dir = "../../../imagenet"
dir_to = "../../../subset100_000"
train_set_len = 100000
validation_set_len = 10000

def check_if_ok(path):
    try:
        img = Image.open(path)

    except (OSError, ValueError, IOError):
        print("Damaged:", path)
        return False

    rgb = np.array(img)
    if len(rgb.shape) == 3 and (rgb.shape[2]) == 3:  # avoid black and white photos
        return True
    else:
        return False


""" This part list the dir and count number of images in subdirectories """

# image_dirs = os.listdir(dataset_dir)
#
# sizes = []
# for i, im_dir in enumerate(image_dirs):
#     sizes.append(len(os.listdir(os.path.join(dataset_dir, im_dir))))
#     if i % 100 == 0:
#         print(i)
#
# if not os.path.isdir(dir_to):
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
if not os.path.isdir(os.path.join(dir_to, "train")):
    os.mkdir(os.path.join(dir_to, "train"))
if not os.path.isdir(os.path.join(dir_to, "validation")):
    os.mkdir(os.path.join(dir_to, "validation"))


# function to copy image
def copy_im(image_dir, goal_dir, h):
    images = os.listdir(os.path.join(dataset_dir, image_dir))
    ch = choice(images)
    if check_if_ok(os.path.join(dataset_dir, d, ch)):
        copyfile(os.path.join(dataset_dir, d, ch), os.path.join(dir_to, goal_dir, ch))
        print(ch, file=h)
        return True
    else:
        return False

t = time.time()

# copy files to dir and write file with names
with open(os.path.join(dir_to, "train.txt"), 'w') as handle:
    count_im = len(os.listdir(os.path.join(dir_to, "train")))
    for d in dir_choices_train:
        if count_im >= train_set_len:
            break
        copy_im(d, "train", handle)
        if count_im % 1000 == 0:
            print(count_im, time.time() - t)
        count_im += 1

    count_im = len(os.listdir(os.path.join(dir_to, "train")))
    # add to match data-set s(ize
    while count_im <= train_set_len:
        d = np.random.choice(image_dirs, 1, p=dir_probabilities)
        if copy_im(str(d[0]), "train", handle):
            if count_im % 1000 == 0:
                print(count_im, time.time() - t)
            count_im += 1

print("validation")
with open(os.path.join(dir_to, "validation.txt"), 'w') as handle:
    count_im = len(os.listdir(os.path.join(dir_to, "validation")))
    for d in dir_choices_validation:
        if count_im > train_set_len:
            break
        copy_im(d, "validation", handle)
        if count_im % 1000 == 0:
            print(count_im, time.time() - t)
        count_im += 1

    count_im = len(os.listdir(os.path.join(dir_to, "validation")))
    # add to match data-set size
    while count_im <= validation_set_len:
        d = np.random.choice(image_dirs, 1, p=dir_probabilities)
        if copy_im(str(d[0]), "validation", handle):
            if count_im % 1000 == 0:
                print(count_im, time.time() - t)
            count_im += 1
