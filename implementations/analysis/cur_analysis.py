import numpy as np
import os
import pickle

import scipy.stats
from Orange.projection import CUR, PCA
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
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

# rank values in M
array = np.array(M)
order = array.argsort(axis=1)
ranks = order.argsort(axis=1)

# print(ranks[:, :10])

rho, p = scipy.stats.spearmanr(M, axis=1)

# print(rho)
print(" ".join(["".ljust(10)] + ["%s" % v[:10] for v in rmse_files]))
for i in range(0, len(rho)):
    print(" ".join([rmse_files[i][:10]] + [("%.4f" % v).ljust(10) for v in rho[i, :]]))

""" pack rho values to orange data table to perform MDS """
domain = Domain([ContinuousVariable(alg) for alg in rmse_files], metas=[StringVariable("Methods")])
table = Table(domain, rho, metas=[[r] for r in rmse_files])
table.save("../../processed_data/spearman_rhos.tab")

""" pack matrix m to table """
domain = Domain([ContinuousVariable(alg) for alg in rmse_files], metas=[StringVariable("Images")])
table = Table(domain, M.T, metas=[[r] for r in images])
table.save("../../processed_data/m_10000_im.tab")

""" Perform CUR """
# data = Table(M)
# # print(data)
# cur = CUR(rank=7, max_error=1, compute_U=False)
# cur_model = cur(data)
#
# transformed_data = cur_model(data, axis=1)
# a = np.argmax(transformed_data, axis=1)
# # print(transformed_data)
#
# # print(a)
#
# """ SVD """
# U, s, Vh = linalg.svd(M, full_matrices=False)
# print(U.shape, s.shape, Vh.shape)
# maxs = np.argmax(Vh, axis=1)
#
# top = Vh.argsort(axis=1)
# # print(top[:, -5:])
#
# selected_images_ids = top[:, -9:].flatten()
# print(len(selected_images_ids))
#
# """ make error matrix for selected """
# A = np.zeros((num_files, len(selected_images_ids)))
# for i, rmse_d in enumerate(rmse_data):
#     rmses_per_method = [rmse_d["rmses"][list(images)[x]] for x in selected_images_ids]
#     A[i, :] = rmses_per_method
#
# A = A.T  # to get methods as a feature
# selected_names = [list(images)[x] for x in selected_images_ids]
# print(len(set(selected_names)))  # check unique
#
# print(A.shape)
#
# """ Pack as a orange table """
# domain = Domain([ContinuousVariable(alg) for alg in rmse_files], metas=[StringVariable("Image")])
# table = Table(domain, A, metas=np.array(selected_names)[:, np.newaxis])
#
# domain_methods = Domain([ContinuousVariable(im) for im in selected_names], metas=[StringVariable("Method")])
# table_methods = Table(domain_methods, A.T, metas=np.array(rmse_files)[:, np.newaxis])
# # print(table.domain)
#
# """ Save data """
# table.save("../../processed_data/top99-images.tab")
# table_methods.save("../../processed_data/top99-methods.tab")