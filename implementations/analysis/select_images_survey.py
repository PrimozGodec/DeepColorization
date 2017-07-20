import os
import numpy as np
from shutil import copyfile

import scipy.misc
from skimage import color

from implementations.support_scripts.image_processing import load_images

dir_from = "../../../validation_colorization-full"
dir_to = "../../../100selection_survey"

# files_from = os.listdir(dir_from)
#
# images = [x.split("-")[-1] for x in files_from]
# images = list(set(images))
#
# methods_order_v = {"hist02-test-full-": 10,
#                   "imp09-test-full-": 4,
#                   "imp9-full-test-full-": 7,
#                   "let-there-color-test-full-": 2}
#
# if not os.path.isdir(dir_to):
#     os.makedirs(dir_to)
#
# for m in methods_order_v.keys():
#     if not os.path.isdir(os.path.join(dir_to, m)):
#         os.makedirs(os.path.join(dir_to, m))
#
# chooses = np.random.choice(images, 100)
#
# for im in chooses:
#     for m in methods_order_v.keys():
#         filename= m + im
#         copyfile(os.path.join(dir_from, filename), os.path.join(dir_to, m,  filename))

# rename selected files

# for directory_ in os.listdir(dir_to):
#     for file_ in os.listdir(os.path.join(dir_to, directory_)):
#         name = file_.split("-")[-1]
#         os.rename(os.path.join(dir_to, directory_, file_), os.path.join(dir_to, directory_, name))

# get originals

# for file_ in os.listdir(os.path.join(dir_to, "imp9-full-test-full-")):
#     copyfile("../../../validation/" + file_,
#              "../../../EvalueationApp/static/images/originals/" + file_)

import matplotlib.pyplot as plt

for file_ in os.listdir("../../../EvalueationApp/static/images/imp09-test-full-"):
    image1 = load_images("../../../EvalueationApp/static/images/imp09-test-full-", file_, size=(224, 224))
    scipy.misc.toimage(color.lab2rgb(image1), cmin=0.0, cmax=1.0)\
        .save(os.path.join("../../../EvalueationApp/static/images/imp09-test-full-", file_))
