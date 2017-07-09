"""
This script will be used to make some nice graphics for a master thesis
"""
import os

import math
from operator import itemgetter

import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import color

from implementations.support_scripts.image_processing import load_images


def blackwhite_colorized_comparison(dir_color):
    images = os.listdir(dir_color)

    # dfine plot
    num_col = 6
    num_rows = math.ceil(len(images) * 2 / num_col)
    plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
    gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                            wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)

    # make plots
    for i, image in enumerate(images):
        image1 = load_images(dir_color, image)

        # plot image
        ax1 = plt.subplot(gs1[i * 2])
        ax1.imshow(image1[:, :, 0], cmap='gray')
        ax2 = plt.subplot(gs1[i * 2 + 1])
        ax2.imshow(color.lab2rgb(image1))
        ax1.axis('off')
        ax2.axis('off')




    plt.savefig("../../../black-colored-comparison.jpg", bbox_inches='tight')
    plt.close()


rename_methods = {"": "Originalna slika",
                  "colorful-test-100": "Zang in sod.",
                  "hist02-test-100": "Klas. brez uteži - plitva arh.",
                  "hist03-test-100": "Klas. brez uteži - globja arh.",
                  "hist04-test-100": "Klas. z utežmi - globja arh.",
                  "hist05-test-100": "Klas. z utežmi - plitva arh.",
                  "hyper03-test-100": "Dahl",
                  "imp09-test-100": "Reg. po delih",
                  "imp9-full-100": "Reg. celotna slika",
                  "imp09-wsm-test-100": "Reg. po delih - brez softmax",
                  "imp10-test-100": "Reg. po delih - brez globalne mreže",
                  "imp10-full-100": "Reg. celotna slika - brez globalne mreže",
                  "let-there-color-test-100": "Iizuka in sod.",
                  "vgg-test-100": "Reg. celotna slika VGG"}

methods_order = {"": 0,
                  "colorful-test-100": 1,
                  "hist02-test-100": 10,
                  "hist03-test-100": 11,
                  "hist04-test-100": 13,
                  "hist05-test-100": 12,
                  "hyper03-test-100": 3,
                  "imp09-test-100": 4,
                  "imp9-full-100": 7,
                  "imp09-wsm-test-100": 5,
                  "imp10-test-100": 6,
                  "imp10-full-100": 8,
                  "let-there-color-test-100": 2,
                  "vgg-test-100": 9}


def alg_comparison(im_dir, methods, images):

    # dfine plot

    num_rows = len(methods)
    num_col = len(images)

    plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
    gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                            wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)

    orders_tup = sorted(methods_order.items(), key=itemgetter(1))

    ordered_methods = [x[0] for x in orders_tup]


    # make plots
    for j in range(num_rows):
        for i in range(num_col):

            image1 = load_images(im_dir, ordered_methods[j] + images[i])

            # plot image
            ax1 = plt.subplot(gs1[j * num_col + i])
            ax1.imshow(color.lab2rgb(image1))

            ax1.axis('off')

            # if ax1.is_first_row():
            #     ax1.set_title(images[i].split(".")[0], fontsize=9)
            if ax1.is_first_col():
                ax1.text(-0.12, 0.5, rename_methods[ordered_methods[j]],
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax1.transAxes,
                         color="black",
                         rotation=90)



    plt.savefig("../../../images-methods-comparison-100.jpg", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # blackwhite_colorized_comparison("../../../colorized_im")


    images_list = ["n09205509_3677.JPEG", "n01850373_5523.JPEG",  # the good ones
                   "n03439814_15864.JPEG", "n01839750_2625.JPEG",  # so so
                   "n11950345_7349.JPEG", "n12953206_10800.JPEG"]  # worst

    alg_comparison("../../../selected-images-100", list(rename_methods.keys()),
                   images_list)