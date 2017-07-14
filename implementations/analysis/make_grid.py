import os

import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg


grid_dir = "../../hist_graphs"
grid_width = 3

images = os.listdir(grid_dir)

rename_methods = {"colorful2": "Zang in sod.",
                  "hist02": "Klas. brez uteži - plitva arh.",
                  "hist03": "Klas. brez uteži - globja arh.",
                  "hist04": "Klas. z utežmi - globja arh.",
                  "hist05": "Klas. z utežmi - plitva arh.",
                  "hyper03": "Dahl",
                  "imp9": "Reg. po delih",
                  "imp9-full": "Reg. celotna slika",
                  "imp9-wsm": "Reg. po delih - brez softmax",
                  "imp10": "Reg. po delih - brez globalne mreže",
                  "imp10-full": "Reg. celotna slika - brez globalne mreže",
                  "let-there-color": "Iizuka in sod.",
                  "vgg1": "Reg. celotna slika VGG"}

order = ["colorful2", "let-there-color", "hyper03", "imp9", "imp9-wsm", "imp10", "imp9-full", "imp10-full",
         "vgg1", "hist02", "hist03", "hist04", "hist05"]

# order = ["let-there-color", "hyper03", "imp9", "imp9-full", "vgg1", "hist02","hist05"]

# dfine plot
num_col = grid_width
num_rows = math.ceil(len(images) / num_col)
plt.figure(figsize=(num_col * 2.5 + 1, (num_rows + 2) * 2.5 + 1))
gs1 = gridspec.GridSpec(num_rows + 2, num_col, width_ratios=[1] * num_col,
                        wspace=0.03, hspace=0.03, top=1, bottom=0, left=0, right=1)

# make plots
for i, image in enumerate(order):
    image1 = mpimg.imread(os.path.join(grid_dir, image + ".jpg"))

    # plot image
    ax1 = plt.subplot(gs1[i])
    ax1.imshow(image1)
    ax1.set_title(rename_methods[image], fontsize=8)
    ax1.axis('off')

plt.savefig("../../result_merged/histograms.pdf", bbox_inches='tight', format='pdf', dpi=900)
plt.close()