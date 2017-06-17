import os

import math
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import gridspec

path_to_photos = "../../result_images"
file_prefix = "imp7d-hist"
plots_per_row = 8
plt.rcParams.update({'font.size': 8})

# remove prefixes
files_with_prefix = [x[len(file_prefix) + 1:] for x in os.listdir(path_to_photos) if x.startswith(file_prefix)]

split_files = [x.split('-') for x in files_with_prefix if len(x.split("-")) == 2]
image_names = sorted(list(set(list(zip(*split_files))[1])))


for image_name in image_names:
    print(image_name)
    # for each image make graphics
    iterations = sorted([x[0] for x in split_files if x[1] == image_name], key=int)
    num_rows = int(math.ceil(len(iterations) / plots_per_row))

    plt.figure(figsize=(plots_per_row * 2.5 + 1, num_rows * 2.5 + 1))
    gs1 = gridspec.GridSpec(num_rows, plots_per_row, width_ratios=[1] * plots_per_row,
         wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)
    # gs1.update(wspace=0, hspace=0)  # set the spacing between axes.

    for i, im in enumerate(iterations):
        # open image
        fname = os.path.join(path_to_photos, file_prefix) + "-" + im + "-" + image_name
        image = Image.open(fname)

        # plot image
        ax1 = plt.subplot(gs1[i])
        ax1.imshow(image)
        # ax1.set_title(int(im) + 5)
        ax1.text(0.03, 0.97, int(im) + 5,
                horizontalalignment='left',
                verticalalignment='top',
                transform = ax1.transAxes,
                 color="white", bbox=dict(boxstyle="round,pad=0.05", fc="black", lw=2))

        ax1.axis('off')


    plt.savefig("../../result_merged/" + file_prefix + "-" + image_name[:-4] + ".pdf", format='pdf', bbox_inches='tight')


