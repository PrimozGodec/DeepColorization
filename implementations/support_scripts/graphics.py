"""
This script will be used to make some nice graphics for a master thesis
"""
import os

import math
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


if __name__ == "__main__":
    blackwhite_colorized_comparison("../../../colorized_im")