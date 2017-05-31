import numpy as np
import scipy.misc
from skimage import color


"""
This script contains functions that are common to all the implementations used
"""
from implementations.support_scripts.image_processing import image_generator_hist, histogram_to_ab


def make_prediction_sample(model, batch_size):
    generator = image_generator_hist(None, "../../test_set", batch_size)  # just generate batch of 10 images
    images_l = next(generator)[0][:1]
    predictions_ab = model.predict(images_l)

    for i in range(images_l.shape[0]):
        # concatenate l and ab for each image
        im = np.concatenate((images_l[i, :, :, :], histogram_to_ab(predictions_ab[i, :, :, :])), axis=2)
        im_rgb = color.lab2rgb(im)
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save('../../result_images/im' + str(i) + '.jpg')
