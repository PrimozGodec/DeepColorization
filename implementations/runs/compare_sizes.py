import os
import sys

from PIL import Image
from skimage import color
import numpy as np

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from implementations.support_scripts.image_processing import load_images_rgb, images_to_l, resize_image
from implementations.support_scripts.image_tester import rmse, psnr
from implementations.models import imp9_32, let_there_color_224, let_there_color_896

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# matrices for multiplying that needs to calculate only once
vec = np.hstack((np.linspace(1/16, 1 - 1/16, 16), np.flip(np.linspace(1/16, 1 - 1/16, 16), axis=0)))
one = np.ones((32, 32))
xv, yv = np.meshgrid(vec, vec)
weight_m = xv * yv
weight_left = np.hstack((one[:, :16], xv[:, 16:])) * yv
weight_right = np.hstack((xv[:, :16], one[:, 16:])) * yv
weight_top = np.vstack((one[:16, :], yv[16:, :])) * xv
weight_bottom = np.vstack((yv[:16, :], one[16:, :])) * xv

weight_top_left = np.hstack((one[:, :16], xv[:, 16:])) * np.vstack((one[:16, :], yv[16:, :]))
weight_top_right = np.hstack((xv[:, :16], one[:, 16:])) * np.vstack((one[:16, :], yv[16:, :]))
weight_bottom_left = np.hstack((one[:, :16], xv[:, 16:])) * np.vstack((yv[:16, :], one[16:, :]))
weight_bottom_right = np.hstack((xv[:, :16], one[:, 16:])) * np.vstack((yv[:16, :], one[16:, :]))


def error_imp9_32(model, name, path, size):
    """
    Used to test let-there-be-color
    """

    # find directory
    test_set_dir_path = path
    image_list = os.listdir(test_set_dir_path)
    num_of_images = len(image_list)

    rmses = []
    psnrs = []
    # repeat for each image
    # lets take first n images
    for i in range(num_of_images):
        # get image
        image_rgb = load_images_rgb(test_set_dir_path, image_list[i], size=size)  # image is of size 256x256
        image_lab = color.rgb2lab(image_rgb)
        image_l = images_to_l(image_lab)

        h, w = image_l.shape

        # split images to list of images
        slices_dim = size[0]//32
        slices = np.zeros((slices_dim * slices_dim * 4, 32, 32, 1))
        for a in range(slices_dim * 2 - 1):
            for b in range(slices_dim * 2 - 1):

                slices[a * slices_dim * 2 + b] = image_l[a*32//2: a*32//2 + 32, b*32//2: b*32//2 + 32, np.newaxis]

        # lover originals dimension to 224x224 to feed vgg and increase dim
        image_l_224_b = resize_image(image_l, (224, 224))
        image_l_224 = np.repeat(image_l_224_b[:, :, np.newaxis], 3, axis=2).astype(float)


        # append together booth lists
        input_data = [slices, np.array([image_l_224,] * slices_dim ** 2 * 4)]

        # predict
        predictions_ab = model.predict(input_data, batch_size=32)

        # reshape back
        original_size_im = np.zeros((h, w, 2))

        for n in range(predictions_ab.shape[0]):
            a, b = n // (slices_dim * 2) * 16, n % (slices_dim * 2) * 16

            if a + 32 > 256 or b + 32 > 256:
                continue  # it is empty edge

            # weight decision
            if a == 0 and b == 0:
                weight = weight_top_left
            elif a == 0 and b == 224:
                weight = weight_top_right
            elif a == 0:
                weight = weight_top
            elif a == 224 and b == 0:
                weight = weight_bottom_left
            elif b == 0:
                weight = weight_left
            elif a == 224 and b == 224:
                weight = weight_bottom_right
            elif a == 224:
                weight = weight_bottom
            elif b == 224:
                weight = weight_right
            else:
                weight = weight_m

            im_a = predictions_ab[n, :, :, 0] * weight
            im_b = predictions_ab[n, :, :, 1] * weight

            original_size_im[a:a+32, b:b+32, :] += np.stack((im_a, im_b), axis=2)

        rmses.append(rmse(original_size_im, image_lab[:, :, 1:]))

        # to rgb
        color_im = np.concatenate((image_l[:, :, np.newaxis], original_size_im), axis=2)
        # color_im = np.concatenate(((np.ones(image_l.shape) * 50)[:, :, np.newaxis], original_size_im), axis=2)
        im_rgb = color.lab2rgb(color_im)

        # calculate psnr
        psnrs.append(psnr(im_rgb * 256, image_rgb))

        # save
        # abs_svave_path = os.path.join(get_abs_path('../../validation_colorization/'))
        # commented to speedup
        # scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + name + image_list[i])

        # print progress
        if i % 500 == 0:
            print(i)

    return np.mean(rmses), np.mean(psnrs)


def error_let_there(model, name, path, im_size):
    """
    Used to test let-there-be-color
    """

    b_size = 32
    im_w, im_h = im_size

    image_list = os.listdir(path)
    num_of_images = len(image_list)

    rmses = []
    psnrs = []

    print("total batches:", num_of_images // b_size)

    for batch_n in range(num_of_images // b_size):
        all_images_l = np.zeros((b_size, im_w, im_h, 1))
        all_images = np.zeros((b_size, im_w, im_h, 3))
        all_images_rgb = np.zeros((b_size, im_w, im_h, 3))
        for i in range(b_size):
            # get image
            image_rgb = load_images_rgb(path, image_list[batch_n * b_size + i], size=im_size)  # image is of size 256x256
            image_lab = color.rgb2lab(image_rgb)
            image_l = images_to_l(image_lab)
            all_images_l[i, :, :, :] = image_l[:, :, np.newaxis]
            all_images[i, :, :, :] = image_lab
            all_images_rgb[i, :, :, :] = image_rgb

        all_vgg = np.zeros((num_of_images, 224, 224, 3))
        for i in range(b_size):
            cur_im = Image.fromarray(all_images_l[i], "LAB")
            all_vgg[i, :, :, :] = np.tile(np.array(cur_im.resize((224, 224), Image.ANTIALIAS)), (1, 1, 1, 3))

        color_im = model.predict([all_images_l, all_vgg], batch_size=b_size)

        rmses += list(rmse(color_im, all_images[:, :, :, 1:]))

        # abs_save_path = get_abs_path('../../validation_colorization/')
        for i in range(b_size):
            # to rgb
            lab_im = np.concatenate((all_images_l[i, :, :, :], color_im[i]), axis=2)
            im_rgb = color.lab2rgb(lab_im)

            # calculate psnr
            psnrs.append(psnr(im_rgb * 256, all_images_rgb[i, :, : :]))

            # save
            # scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_save_path + name + image_list[batch_n * b_size + i])
        print(batch_n)

    return np.mean(rmses), np.mean(psnrs)


b_size = 32

# load all three models
model_imp9 = imp9_32.model()
model_let_224 = let_there_color_224.model()
model_let_896 = let_there_color_896.model()

model_imp9.load_weights("../../weights/implementation9-12.h5")
model_let_224.load_weights("../../weights/let-there-color-2.h5")
model_let_896.load_weights("../../weights/let-there-color-2.h5")

images_dir = "../../../big_dataset"

# test imp9
# print(error_let_there(model_let_224, "test", images_dir, im_size=(224, 224)))
print(error_let_there(model_let_896, "test", images_dir, im_size=(896, 896)))
print(error_imp9_32(model_imp9, "test", images_dir, size=(224, 224)))
print(error_imp9_32(model_imp9, "test", images_dir, size=(896, 896)))