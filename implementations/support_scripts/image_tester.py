import inspect
import os

import scipy
from skimage import color
import scipy.misc

import numpy as np

from implementations.support_scripts.image_processing import load_images, images_to_l, resize_image, load_images_rgb


def get_abs_path(relative):
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
    return os.path.join(script_dir, relative)


def rmse(im1, im2):
    RMSE = np.mean((im1 - im2) ** 2, axis=(-1, -2, -3)) ** 0.5
    return RMSE


def psnr(im1, im2):
    PSNR = 20 * np.log10(255 / np.mean((im1 - im2) ** 2, axis=(-1, -2, -3)) ** 0.5)
    return PSNR


def image_error_full_vgg(model, name, b_size=32):
    """
    Used to test let-there-be-color
    """

    abs_file_path = get_abs_path("../../../subset100_000/validation")
    image_list = os.listdir(abs_file_path)
    num_of_images = len(image_list)

    rmses = []
    psnrs = []

    print("total batches:", num_of_images // b_size)

    for batch_n in range(num_of_images // b_size):
        all_images_l = np.zeros((b_size, 224, 224, 1))
        all_images = np.zeros((b_size, 224, 224, 3))
        all_images_rgb = np.zeros((b_size, 224, 224, 3))
        for i in range(b_size):
            # get image
            image_rgb = load_images_rgb(abs_file_path, image_list[batch_n * b_size + i], size=(224, 224))  # image is of size 256x256
            image_lab = color.rgb2lab(image_rgb)
            image_l = images_to_l(image_lab)
            all_images_l[i, :, :, :] = image_l[:, :, np.newaxis]
            all_images[i, :, :, :] = image_lab
            all_images_rgb[i, :, :, :] = image_rgb

        all_vgg = np.zeros((num_of_images, 224, 224, 3))
        for i in range(b_size):
            all_vgg[i, :, :, :] = np.tile(all_images_l[i], (1, 1, 1, 3))

        color_im = model.predict([all_images_l, all_vgg], batch_size=b_size)

        rmses += list(rmse(color_im, all_images[:, :, :, 1:]))

        abs_save_path = get_abs_path('../../validation_colorization/')
        for i in range(b_size):
            # to rgb
            lab_im = np.concatenate((all_images_l[i, :, :, :], color_im[i]), axis=2)
            im_rgb = color.lab2rgb(lab_im)

            # calculate psnr
            psnrs.append(psnr(im_rgb * 256, all_images_rgb[i, :, : :]))

            # save
            scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_save_path + name + image_list[batch_n * b_size + i])
        print(batch_n)

    print("RMSE:", np.mean(rmses))
    print("PSNR:", np.mean(psnrs))


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


def image_error_small_vgg(model, name):
    """
    Used to test let-there-be-color
    """

    # find directory
    test_set_dir_path = get_abs_path("../../../subset100_000/validation")
    image_list = os.listdir(test_set_dir_path)
    num_of_images = len(image_list)

    rmses = []
    psnrs = []
    # repeat for each image
    # lets take first n images
    for i in range(num_of_images):
        # get image
        image_rgb = load_images_rgb(test_set_dir_path, image_list[i])  # image is of size 256x256
        image_lab = color.rgb2lab(image_rgb)
        image_l = images_to_l(image_lab)

        h, w = image_l.shape

        # split images to list of images
        slices_dim = 256//32
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
        abs_svave_path = os.path.join(get_abs_path('../../validation_colorization/'))
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + name + image_list[i])

        # print progress
        if i % 500 == 0:
            print(i)

    print("RMSE:", np.mean(rmses))
    print("PSNR:", np.mean(psnr))


def image_error_vgg(model, name, b_size=32, dim=3):
    test_set_dir_path = get_abs_path("../../../subset100_000/validation")
    image_list = os.listdir(test_set_dir_path)
    num_of_images = len(image_list)

    rmses = []
    psnrs = []

    for batch_n in range(num_of_images // b_size):
        all_images = np.zeros((b_size, 224, 224, 3))
        all_images_rgb = np.zeros((b_size, 224, 224, 3))
        all_images_l = np.zeros((b_size, 224, 224, dim))
        for i in range(b_size):
            # get image
            image_rgb = load_images_rgb(test_set_dir_path, image_list[batch_n * b_size + i], size=(224, 224))
            image_lab = color.rgb2lab(image_rgb)
            all_images[i, :, :, :] = image_lab
            image_l = images_to_l(image_lab)
            all_images_l[i, :, :, :] = np.tile(image_l[:, :, np.newaxis], (1, 1, 1, dim))
            all_images_rgb[i, :, :, :] = image_rgb


        color_im = model.predict(all_images_l, batch_size=b_size)

        rmses += list(rmse(color_im, all_images[:, :, :, 1:]))

        abst_path = get_abs_path('../../validation_colorization/')
        for i in range(b_size):
            # to rgb
            lab_im = np.concatenate((all_images_l[i, :, :, 0][:, :, np.newaxis], color_im[i]), axis=2)
            im_rgb = color.lab2rgb(lab_im)

            # psnr
            psnrs.append(psnr(im_rgb * 256, all_images_rgb[i, :, :, :]))

            # save
            scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abst_path + name + image_list[batch_n * b_size + i])
        print(batch_n)

    print("RMSE:", np.mean(rmses))
    print("PSNR:", np.mean(psnrs))


def image_error_small_hist(model, name):
    test_set_dir_path = get_abs_path("../../../subset100_000/validation")
    # find directory
    image_list = os.listdir(test_set_dir_path)
    num_of_images = len(image_list)

    rmses = []
    psnrs = []

    # repeat for each image
    # lets take first n images
    for i in range(num_of_images):
        # get image
        image_rgb = load_images_rgb(test_set_dir_path, image_list[i])  # image is of size 256x256
        image_lab = color.rgb2lab(image_rgb)
        image_l = images_to_l(image_lab)

        h, w = image_l.shape

        # split images to list of images
        slices_dim = 256//32
        slices = np.zeros((slices_dim * slices_dim * 4, 32, 32, 1))
        for a in range(slices_dim * 2 - 1):
            for b in range(slices_dim * 2 - 1):
                slices[a * slices_dim * 2 + b] = image_l[a*32//2: a*32//2+32, b*32//2: b*32//2+32, np.newaxis]

        # lover originals dimension to 224x224 to feed vgg and increase dim
        image_l_224_b = resize_image(image_l, (224, 224))
        image_l_224 = np.repeat(image_l_224_b[:, :, np.newaxis], 3, axis=2).astype(float)


        # append together booth lists
        input_data = [slices, np.array([image_l_224,] * slices_dim ** 2 * 4)]

        # predict
        predictions_hist = model.predict(input_data)

        # reshape back
        indices = np.argmax(predictions_hist[:, :, :, :], axis=3)

        predictions_a = indices // 20 * 10 - 100 + 5
        predictions_b = indices % 20 * 10 - 100 + 5  # +5 to set in the middle box

        predictions_ab = np.stack((predictions_a, predictions_b), axis=3)
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

        psnrs.append(psnr(im_rgb * 256, image_rgb))

        # save
        abs_svave_path = os.path.join(get_abs_path('../../validation_colorization/'))
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + name + image_list[i])

        if i % 500 == 0:
            print(i)

    print("RMSE:", np.mean(rmses))
    print("PSNR:", np.mean(psnrs))


def image_error_hist(model, name, b_size=32):
    test_set_dir_path = get_abs_path("../../../subset100_000/validation")
    image_list = os.listdir(test_set_dir_path)
    num_of_images = len(image_list)

    rmses = []
    psnrs = []

    for batch_n in range(num_of_images // b_size):
        all_images = np.zeros((b_size, 224, 224, 3))
        all_images_rgb = np.zeros((b_size, 224, 224, 3))
        all_images_l = np.zeros((b_size, 224, 224, 1))
        for i in range(b_size):
            # get image
            image_rgb = load_images_rgb(test_set_dir_path, image_list[batch_n * b_size + i], size=(224, 224))  # image is of size 256x256
            image_lab = color.rgb2lab(image_rgb)
            all_images[i, :, :, :] = image_lab
            image_l = images_to_l(image_lab)
            all_images_l[i, :, :, :] = image_l[:, :, np.newaxis]
            all_images_rgb[i, :, :, :] = image_rgb

        color_im = model.predict(all_images_l, batch_size=b_size)

        for i in range(b_size):
            # to rgb
            idx = np.argmax(color_im[i], axis=2)
            a = idx // 20 * 10.0 - 100 + 5
            b = idx % 20 * 10.0 - 100 + 5
            lab_im = np.concatenate((all_images[i, :, :, 0][:, :, np.newaxis],
                                     a[:, :, np.newaxis], b[:, :, np.newaxis]), axis=2)
            im_rgb = color.lab2rgb(lab_im)

            rmses.append(rmse(lab_im[:, :, 1:], all_images[i, :, :, 1:]))
            psnrs.append(psnr(im_rgb * 256, all_images_rgb[i, :, :, :]))

            # save
            abs_svave_path = os.path.join(get_abs_path('../../validation_colorization/'))
            scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + name + image_list[batch_n * b_size + i])

    print("RMSE:", np.mean(rmses))
    print("PSNR:", np.mean(psnrs))