import numpy as np
import os
import pickle
from Orange.projection import CUR, PCA
from Orange.data import Table, Domain, ContinuousVariable
from scipy import linalg

""" Construct RMSE matrix M """

rmse_files = os.listdir("../../rmses")
num_files = len(rmse_files)

rmse_data = []

# read all files
for rmse_file in rmse_files:
    with (open(os.path.join("../../rmses", rmse_file), "rb")) as f:
        rmse_data.append(pickle.load(f))

images = rmse_data[0]["rmses"].keys()

M = np.zeros((num_files, len(images)))
for i, rmse_d in enumerate(rmse_data):
    rmses_per_method = [rmse_d["rmses"][x] for x in images]
    M[i, :] = rmses_per_method

""" Perform CUR """
data = Table(M)
# print(data)
cur = CUR(rank=7, max_error=1, compute_U=False)
cur_model = cur(data)

transformed_data = cur_model(data, axis=1)
a = np.argmax(transformed_data, axis=1)
# print(transformed_data)

# print(a)

""" SVD """
U, s, Vh = linalg.svd(M, full_matrices=False)
print(U.shape, s.shape, Vh.shape)
maxs = np.argmax(Vh, axis=1)

top = Vh.argsort(axis=1)
# print(top[:, -5:])

selected_images_ids = top[:, -9:].flatten()
print(len(selected_images_ids))

""" make error matrix for selected """
A = np.zeros((num_files, len(selected_images_ids)))
for i, rmse_d in enumerate(rmse_data):
    rmses_per_method = [rmse_d["rmses"][list(images)[x]] for x in selected_images_ids]
    A[i, :] = rmses_per_method

A = A.T  # to get methods as a feature

print(A.shape)

""" Pack as a orange table """
domain = Domain([ContinuousVariable(alg) for alg in rmse_files])
table = Table(domain, A)
# print(table.domain)

""" Save data """
table.save("../../processed_data/top99.tab")