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



    plt.subplot_tool()
    plt.savefig("../../../black-colored-comparison.jpg", bbox_inches='tight')
    plt.close()


rename_methods = {"": "Originalna slika",
                  "colorful-test-100": "Zang in sod.",
                  # "hist02-test-100": "Klas. brez uteži\nplitva arh.",
                  # "hist03-test-100": "Klas. brez uteži\nglobja arh.",
                  # "hist04-test-100": "Klas. z utežmi\nglobja arh.",
                  # "hist05-test-100": "Klas. z utežmi\nplitva arh.",
                  "hyper03-test-100": "Dahl",
                  "imp09-test-100": "Reg. po delih",
                  "imp9-full-100": "Reg. cel. slika",
                  "imp09-wsm-test-100": "Reg. po delih\nbrez softmax",
                  "imp10-test-100": "Reg. po delih\nbrez glob. mr.",
                  "imp10-full-100": "Reg. cel. slika\nbrez glob. mr.",
                  "let-there-color-test-100": "Iizuka in sod.",
                  "vgg-test-100": "Reg. cel.\nslika VGG"}


methods_order = {"": 0,
                  "colorful-test-100": 1,
                  # "hist02-test-100": 10,
                  # "hist03-test-100": 11,
                  # "hist04-test-100": 13,
                  # "hist05-test-100": 12,
                  "hyper03-test-100": 3,
                  "imp09-test-100": 4,
                  "imp9-full-100": 7,
                  "imp09-wsm-test-100": 5,
                  "imp10-test-100": 6,
                  "imp10-full-100": 8,
                  "let-there-color-test-100": 2,
                  "vgg-test-100": 9}


methods_order_v = {"": 0,
                  # "colorful-test-100": 1,
                  "hist02-test-full-": 10,
                  # "hist03-test-100": 11,
                  # "hist04-test-100": 13,
                  "hist05-test-full-": 12,
                  "hyper03-test-full-": 3,
                  "imp09-test-full-": 4,
                  "imp9-full-test-full-": 7,
                  # "imp09-wsm-test-100": 5,
                  # "imp10-test-100": 6,
                  # "imp10-full-100": 8,
                  "let-there-color-test-full-": 2,
                  "vgg-test-full-": 9}

rename_methods_v = {"": "Originalna slika",
                  "colorful-test-100": "Zang in sod.",
                  "hist02-test-full-": "Klas. brez uteži\nplitva arh.",
                  "hist03-test-100": "Klas. brez uteži\nglobja arh.",
                  "hist04-test-100": "Klas. z utežmi\nglobja arh.",
                  "hist05-test-full-": "Klas. z utežmi\nplitva arh.",
                  "hyper03-test-full-": "Dahl",
                  "imp09-test-full-": "Reg. po delih",
                  "imp9-full-test-full-": "Reg. cel. slika",
                  "imp09-wsm-test-100": "Reg. po delih\nbrez softmax",
                  "let-there-color-test-full-": "Iizuka in sod.",
                  "vgg-test-full-": "Reg. celotna\nslika VGG"}


plt.rcParams.update({'font.size': 22})

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
                ax1.text(-0.10, 0.5, rename_methods[ordered_methods[j]],
                         horizontalalignment='right',
                         verticalalignment='center',
                         transform=ax1.transAxes,
                         color="black",
                         rotation=90,
                         multialignment='center')



    plt.savefig("../../../images-methods-comparison-100.jpg", bbox_inches='tight')
    plt.close()


def alg_comparison_vertical(im_dir):


    # dfine plot
    files = os.listdir(im_dir)
    just_im_names = list(set([x.split("-")[-1] for x in files]))
    # just_im_names = ['n02215770_4433.JPEG', 'Screen Shot 2017-07-05 at 12.42.28.png', 'n02867401_971.JPEG', 'n02783994_7940.JPEG', 'n03219483_4276.JPEG', 'n03891538_27243.JPEG', 'n02213543_4098.JPEG', 'n01322221_4873.JPEG', 'n02903126_318.JPEG', 'n02124623_8147.JPEG', 'maruti-suzuki-swift-default-image.png-version201707131518.png', 'n02940385_203.JPEG']

    num_col = 8
    num_rows = len(just_im_names)

    plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
    gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                            wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)

    orders_tup = sorted(methods_order_v.items(), key=itemgetter(1))

    ordered_methods = [x[0] for x in orders_tup]


    # make plots
    for j in range(num_rows):
        for i in range(num_col):

            image1 = load_images(im_dir, ordered_methods[i] + just_im_names[j])

            # plot image
            ax1 = plt.subplot(gs1[j * num_col + i])
            ax1.imshow(color.lab2rgb(image1))

            ax1.axis('off')

            # if ax1.is_first_row():
            #     ax1.set_title(images[i].split(".")[0], fontsize=9)
            if ax1.is_first_row():
                ax1.set_title(rename_methods_v[ordered_methods[i]], fontsize=22, y=1.08)
                # ax1.text(-0.12, 0.5, rename_methods_v[ordered_methods[i]],
                #          horizontalalignment='left',
                #          verticalalignment='center',
                #          transform=ax1.transAxes,
                #          color="black",
                #          rotation=90)



    plt.savefig("../../../images-methods-comparison-full.pdf", bbox_inches='tight', format='pdf')
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


def make_grid(num_col, num_rows, image_dir):

    plt.figure(figsize=(num_col * 2.5 + 0.5, (num_rows + 2) * 2.5 + 2.1))
    gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                            wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)

    ims = sorted(os.listdir(image_dir))
    for i, im in enumerate(ims):
        image1 = mpimg.imread(os.path.join(image_dir, im))

        # plot image
        ax1 = plt.subplot(gs1[i])
        ax1.imshow(image1, cmap="gray")
        ax1.set_title("Nivo " + str(i + 1), fontsize=7)

        ax1.axis('off')

    plt.savefig("../../../visualisations_merged/complete.jpg", bbox_inches='tight', dpi=1200)
    plt.close()



if __name__ == "__main__":
    blackwhite_colorized_comparison("../../../colorized_im")


    # images_list = ["n09205509_3677.JPEG", "n01850373_5523.JPEG",  # the good ones
    #                "n03439814_15864.JPEG", "n01839750_2625.JPEG",  # so so
    #                "n11950345_7349.JPEG", "n12953206_10800.JPEG"]  # worst
    #
    # alg_comparison("../../../selected-images-100", list(rename_methods.keys()),
    #                images_list)

    # visualize_activations("../../../visualisations/")
    # make_grid(2, 3, "../../../visualisations_merged")

    alg_comparison_vertical("../../../selection-full/dobre")
    # blackwhite_colorized_comparison("../../../old-colorized")