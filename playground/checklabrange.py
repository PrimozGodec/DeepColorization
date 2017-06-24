import numpy as np
from skimage import color

values = []

for r in range(256):
     print(r)
     for g in range(256):
        for b in range(256):
            values.append(color.rgb2lab([[[r, g, b]]])[0, 0, :])

v = np.vstack(values)
print(np.min(v, axis=1))
print(np.max(v, axis=1))

