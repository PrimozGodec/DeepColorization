import os
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve


def compute_color_prior(size=64, do_plot=False):

    with h5py.File(os.path.join(data_dir, "images_%s_data.h5" % size), "a") as hf:
        # Compute the color prior over a subset of the training set
        # Otherwise it is quite long
        X_ab = hf["train_lab_data"][:100][:, 1:, :, :]
        npts, c, h, w = X_ab.shape
        X_a = np.ravel(X_ab[:, :, :, 0])
        X_b = np.ravel(X_ab[:, :, :, 1])
        X_ab = np.vstack((X_a, X_b)).T

        print(np.min(X_a))
        print(np.max(X_b))

        ind = ((X_a + 100) / 10).astype(int) * 20 + ((X_b + 100) / 10).astype(int)
        print(np.min(ind))
        print(np.max(ind))
        # We now count the number of occurrences of each color
        ind = np.ravel(ind)
        prior_prob = np.bincount(ind, minlength=400)
        print(prior_prob)

        # We turn this into a color probability
        prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

        # Save
        np.save(os.path.join(data_dir, "images_%s_prior_prob.npy" % size), prior_prob)


def smooth_color_prior(size=64, sigma=5, do_plot=False):

    prior_prob = np.load(os.path.join(data_dir, "images_%s_prior_prob.npy" % size))
    # add an epsilon to prior prob to avoid 0 vakues and possible NaN
    prior_prob += 1E-3 * np.min(prior_prob)
    # renormalize
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Smooth with gaussian
    yy = prior_prob.reshape((20, 20))
    window = gaussian(2000, sigma)  # 2000 pts in the window, sigma=5
    smoothed = convolve(yy, window / window.sum(), mode='same')

    prior_prob_smoothed = smoothed.ravel()
    prior_prob_smoothed = prior_prob_smoothed / np.sum(prior_prob_smoothed)

    # Save
    file_name = os.path.join(data_dir, "images_%s_prior_prob_smoothed.npy" % size)
    np.save(file_name, prior_prob_smoothed)


def compute_prior_factor(size=64, gamma=0.5, alpha=1, do_plot=False):

    file_name = os.path.join(data_dir, "images_%s_prior_prob_smoothed.npy" % size)
    prior_prob_smoothed = np.load(file_name)

    u = np.ones_like(prior_prob_smoothed)
    u = u / np.sum(1.0 * u)

    prior_factor = (1 - gamma) * prior_prob_smoothed + gamma * u
    prior_factor = np.power(prior_factor, -alpha)

    # renormalize
    prior_factor = prior_factor / (np.sum(prior_factor * prior_prob_smoothed))

    file_name = os.path.join(data_dir, "images_%s_prior_factor.npy" % size)
    np.save(file_name, prior_factor)


data_dir = "../../../subset100_000"
compute_color_prior(size=256)
smooth_color_prior(size=256)
compute_prior_factor(size=256)