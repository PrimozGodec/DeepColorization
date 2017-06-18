import os

import math
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import gridspec

path_to_photos = "../../result_images"
# file_prefix = ["imp7d", "imp7d-relu", "imp7d-reg", "imp7d-hist", "imp7d-01"]
file_prefix = ["imp8", "imp8-deep", "imp8-res", "imp8-pool", "imp8-poola", "imp8-trans", "imp8-bn"]

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

    num_rows = len(it)
    num_col = len(file_prefix)

    plt.figure(figsize=(num_col * 2.5 + 1, num_rows * 2.5 + 1))
    gs1 = gridspec.GridSpec(num_rows, num_col, width_ratios=[1] * num_col,
         wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)

    for i, im in enumerate(it):
        # open image
        for j, alg in enumerate(file_prefix):
            fname = os.path.join(path_to_photos, alg) + "-" + im + "-" + image_name
            image = Image.open(fname)

            # plot image
            ax1 = plt.subplot(gs1[i * num_col + j])
            ax1.imshow(image)

            ax1.text(0.03, 0.97, int(im) + 5,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform = ax1.transAxes,
                     color="white", bbox=dict(boxstyle="round,pad=0.05", fc="black", lw=2))

            ax1.axis('off')
            if ax1.is_first_row():
                ax1.set_title(alg, fontsize=9)


    plt.savefig(
        "../../result_merged/" + image_name[:-4] + "-" + "_".join(file_prefix) + ".pdf",
        format='pdf', bbox_inches='tight')


