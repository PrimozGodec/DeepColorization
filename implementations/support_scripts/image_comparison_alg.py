import os

import math
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import gridspec
from skimage import color

from implementations.support_scripts.image_processing import load_images

path_to_photos = "../../result_images"
# file_prefix = ["imp7d", "imp7d-relu", "imp7d-reg", "imp7d-hist", "imp7d-01"]
file_prefix = ["let-there-color", "hyper03", "imp9", "imp9-full", "vgg1", "hist02", "hist05"]

rename_methods = {"colorful": "Zang in sod.",
                  "hist02": "Klas. brez uteži - plitva arh.",
                  "hist03": "Klas. brez uteži - arih. 2",
                  "hist04": "Klas. z utežmi - arih. 2",
                  "hist05": "Klas. z utežmi - plitva arh.",
                  "hyper03": "Dahl",
                  "imp9": "Reg. po delih",
                  "imp9-full": "Reg. celotna slika",
                  "imp10": "Reg. po delih - brez globalne mreže",
                  "let-there-color": "Iizuka in sod.",
                  "vgg1": "Reg. celotna slika VGG"}

plt.rcParams.update({'font.size': 8})

# select photos that exist in all algorithms
im_names = []
split_names = []
for alg in file_prefix:
    files_with_prefix = [x[len(alg) + 1:] for x in os.listdir(path_to_photos) if x.startswith(alg)]

    split_files = [x.split('-') for x in files_with_prefix if len(x.split("-")) == 2]
    split_names.append(split_files)

    image_names = sorted(list(set(list(zip(*split_files))[1])))
    im_names.append(image_names)

im_names = list(map(set, im_names))
im_names_all = sorted(list(set.intersection(*im_names)))

im_names_all = list(im_names_all)

# plot for each image
for image_name in im_names_all:
    print(image_name)
    # for each image make graphics
    it = []
    for sp in split_names:
        it.append([x[0] for x in sp if x[1] == image_name])
    it = list(map(set, it))
    it = sorted(list(set.intersection(*it)), key=int)

    num_rows = 10 # len(it) // 5
    num_col = len(file_prefix)

    plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
    gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
         wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)

    original = []

    # add grayscale image at beginning
    for j, alg in enumerate(file_prefix):
        image = load_images("../../test_set", image_name)
        # original.append(image)

        # plot image
        ax1 = plt.subplot(gs1[j])
        ax1.imshow(image[:, :, 0], cmap='gray')

        ax1.text(0.03, 0.97, "črno-bela",
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax1.transAxes,
                 color="white", bbox=dict(boxstyle="round,pad=0.05", fc="black", lw=2))
        ax1.axis('off')
        if ax1.is_first_row():
            ax1.set_title(rename_methods[alg], fontsize=9)


        ax1 = plt.subplot(gs1[- j - 1])
        ax1.imshow(color.lab2rgb(image[:, :, :]))

        ax1.text(0.03, 0.97, "originalna",
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax1.transAxes,
                 color="white", bbox=dict(boxstyle="round,pad=0.05", fc="black", lw=2))

        ax1.axis('off')


    for i, im in enumerate(it):
        # if (int(im) + 1) % 5 != 0:
        #     continue

        if i >= 10:
            break

        # open image
        for j, alg in enumerate(file_prefix):
            fname = os.path.join(path_to_photos, alg) + "-" + im + "-" + image_name
            image = Image.open(fname)

            # plot image
            ax1 = plt.subplot(gs1[i * num_col + j + num_col])
            ax1.imshow(image)

            ax1.text(0.03, 0.97, int(im) + 1,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform = ax1.transAxes,
                     color="white", bbox=dict(boxstyle="round,pad=0.05", fc="black", lw=2))

            ax1.axis('off')
            if ax1.is_first_row():
                ax1.set_title(alg, fontsize=9)

            image.close()


    plt.savefig(
        "../../result_merged/" + image_name[:-4] + "-" + "_".join(file_prefix) + ".pdf",
        format='pdf', bbox_inches='tight')
    plt.close()

