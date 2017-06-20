import os

import pickle

dataset_dir = "../../../imagenet"
dir_to = "../../../subset100_000"

""" This part list the dir and count number of images in subdirectories """

image_dirs = os.listdir(dataset_dir)

sizes = []
for i, im_dir in image_dirs:
    sizes.append(len(os.listdir(os.path.join(dataset_dir, im_dir))))
    if i % 100 == 0:
        print(i)

file_name = os.path.join(dir_to, "file_dist")

with open(file_name, 'wb') as handle:
    pickle.dump((image_dirs, sizes), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(file_name, 'rb') as handle:
    image_dirs, sizes = pickle.load(handle)

print(sum(sizes))