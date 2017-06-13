from implementations.support_scripts.common import h5_small_vgg_generator
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt


g = h5_small_vgg_generator(16, "../h5_data", None)
b = next(g)
vgg = b[0][1][4, :, :, :]
s = b[0][0][4, :, :, :]
c = b[1][4, :, :, :]
im = np.concatenate((s, c), axis=2)
rgb = color.lab2rgb(im)
imgplot = plt.imshow(rgb)
plt.show()
plt.imshow(vgg[:, :, 0])
plt.show()