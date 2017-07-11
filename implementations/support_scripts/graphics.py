"""
This script will be used to make some nice graphics for a master thesis
"""
import os

import math
from operator import itemgetter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def visualize_activations(act_dir):
    files = os.listdir(act_dir)

    split_files = [x.split("_") for x in files]

    images_names = list(set([x[2] + "_" + x[3] for x in split_files]))
    print(len(images_names))
    for image in images_names:

        #1st layer
        print(1)
        im_per_dim = int(math.sqrt(64))
        num_rows = im_per_dim
        num_col = im_per_dim

        plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
        gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                                wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)
        for l1 in range(64):

            image1 = mpimg.imread(os.path.join(act_dir, "0_" + str(l1) + "_" + image))

            # plot image
            ax1 = plt.subplot(gs1[l1])
            ax1.imshow(image1, cmap="gray")

            ax1.axis('off')

        plt.savefig("../../../visualisation_merged/0_" + image, bbox_inches='tight')
        plt.close()

        # 2nd layer
        print(2)
        im_per_dim = int(math.sqrt(128))
        num_rows = im_per_dim
        num_col = im_per_dim

        plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
        gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                                wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)
        for l1 in range(im_per_dim ** 2):
            image1 = mpimg.imread(os.path.join(act_dir, "1_" + str(l1) + "_" + image))

            # plot image
            ax1 = plt.subplot(gs1[l1])
            ax1.imshow(image1, cmap="gray")

            ax1.axis('off')

        plt.savefig("../../../visualisation_merged/1_" + image, bbox_inches='tight')
        plt.close()

        # 3rd layer
        print(3)
        im_per_dim = int(math.sqrt(128))
        num_rows = im_per_dim
        num_col = im_per_dim

        plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
        gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                                wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)
        for l1 in range(im_per_dim ** 2):
            image1 = mpimg.imread(os.path.join(act_dir, "2_" + str(l1) + "_" + image))

            # plot image
            ax1 = plt.subplot(gs1[l1])
            ax1.imshow(image1, cmap="gray")

            ax1.axis('off')

        plt.savefig("../../../visualisation_merged/2_" + image, bbox_inches='tight')
        plt.close()

        # 4rd layer
        print(4)
        im_per_dim = int(math.sqrt(128))
        num_rows = im_per_dim
        num_col = im_per_dim

        plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
        gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                                wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)
        for l1 in range(im_per_dim ** 2):

            image1 = mpimg.imread(os.path.join(act_dir, "3_" + str(l1) + "_" + image))

            # plot image
            ax1 = plt.subplot(gs1[l1])
            ax1.imshow(image1, cmap="gray")

            ax1.axis('off')

        plt.savefig("../../../visualisation_merged/3_" + image, bbox_inches='tight')
        plt.close()

        # 5nd layer
        print(5)
        im_per_dim = int(math.sqrt(128))
        num_rows = im_per_dim
        num_col = im_per_dim

        plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
        gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                                wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)
        for l1 in range(im_per_dim ** 2):

            image1 = mpimg.imread(os.path.join(act_dir, "4_" + str(l1) + "_" + image))

            # plot image
            ax1 = plt.subplot(gs1[l1])
            ax1.imshow(image1, cmap="gray")

            ax1.axis('off')

        plt.savefig("../../../visualisation_merged/4_" + image, bbox_inches='tight')
        plt.close()

        # 6nd layer
        print(6)
        im_per_dim = int(math.sqrt(256))
        num_rows = im_per_dim
        num_col = im_per_dim

        plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
        gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                                wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)
        for l1 in range(im_per_dim ** 2):

            image1 = mpimg.imread(os.path.join(act_dir, "5_" + str(l1) + "_" + image))

            # plot image
            ax1 = plt.subplot(gs1[l1])
            ax1.imshow(image1, cmap="gray")

            ax1.axis('off')

        plt.savefig("../../../visualisation_merged/5_" + image, bbox_inches='tight')
        plt.close()




if __name__ == "__main__":
    # blackwhite_colorized_comparison("../../../colorized_im")


    # images_list = ["n09205509_3677.JPEG", "n01850373_5523.JPEG",  # the good ones
    #                "n03439814_15864.JPEG", "n01839750_2625.JPEG",  # so so
    #                "n11950345_7349.JPEG", "n12953206_10800.JPEG"]  # worst
    #
    # alg_comparison("../../../selected-images-100", list(rename_methods.keys()),
    #                images_list)
    visualize_activations("../../../visualisations/")