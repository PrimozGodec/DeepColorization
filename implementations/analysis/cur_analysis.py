import numpy as np
import os
import pickle

import scipy.stats
from Orange.projection import CUR, PCA
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from scipy import linalg

""" methods names """

rename_methods = {"colorful-test-100": "Zang in sod.",
                  "hist02-test-100": "Klas. brez uteži - arih. 1",
                  "hist03-test-100": "Klas. brez uteži - arih. 2",
                  "hist04-test-100": "Klas. z utežmi - arih. 2",
                  "hist05-test-100": "Klas. z utežmi - arih. 1",
                  "hyper03-test-100": "Dahl",
                  "imp09-test-100": "Reg. po delih",
                  "imp9-full-100": "Reg. celotna slika",
                  "imp09-wsm-test-100": "Reg. po delih - brez softmax",
                  "imp10-test-100": "Reg. po delih - brez globalne mreže",
                  "imp10-full-100": "Reg. celotna slika - brez globalne mreže",
                  "let-there-color-test-100": "Iizuka in sod.",
                  "vgg-test-100": "Reg. celotna slika VGG"}

""" Construct RMSE matrix M """

rmse_files = os.listdir("../../rmses")
num_files = len(rmse_files)

# make rigth order
rmse_files = [
    'colorful-test-100.pkl',
    'let-there-color-test-100.pkl',
    'hyper03-test-100.pkl',
    'imp09-test-100.pkl',
    'imp09-wsm-test-100.pkl',
    'imp10-test-100.pkl',
    'imp9-full-100.pkl',
    'imp10-full-100.pkl',
    'vgg-test-100.pkl',
    'hist02-test-100.pkl',
    'hist03-test-100.pkl',
    'hist04-test-100.pkl',
    'hist05-test-100.pkl']

rmse_data = []


# read all files
for rmse_file in rmse_files:
    with (open(os.path.join("../../rmses", rmse_file), "rb")) as f:
        rmse_data.append(pickle.load(f))

rmse_files = [rename_methods[x[:-4]] for x in rmse_files]
print(rmse_files)

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

rho, p = scipy.stats.spearmanr(ranks, axis=1)

# print(rho)
print("&".join(["".ljust(10)] + ["%s" % v[:10].ljust(10) for v in rmse_files]))
for i in range(0, len(rho)):
    print("&".join([rmse_files[i][:10].ljust(10)] + [("%.4f" % v).ljust(10) for v in rho[i, :]]), "\\\\")

# """ pack rho values to orange data table to perform MDS """
# domain = Domain([ContinuousVariable(alg) for alg in rmse_files], metas=[StringVariable("Methods")])
# table = Table(domain, rho, metas=[[r] for r in rmse_files])
# table.save("../../processed_data/spearman_rhos.tab")
#
# """ pack matrix m to table """
# domain = Domain([ContinuousVariable(alg) for alg in rmse_files], metas=[StringVariable("Images")])
# table = Table(domain, M.T, metas=[[r] for r in images])
# table.save("../../processed_data/m_10000_im.tab")
#
# """ pack rank m to table """
# domain = Domain([ContinuousVariable(alg) for alg in rmse_files], metas=[StringVariable("Images")])
# table = Table(domain, ranks.T, metas=[[r] for r in images])
# table.save("../../processed_data/ranks_1000.tab")

""" Perform CUR """
data = Table(ranks)
# print(data)
cur = CUR(rank=3, compute_U=False)
cur_model = cur(data)

transformed_data = cur_model(data, axis=1)
transformed_data = np.array(transformed_data)

find_per_row = 100 // transformed_data.shape[0]
print(transformed_data.shape[0])
print(find_per_row)
all_rows = range(transformed_data.shape[0])
tops = []
for row in range(transformed_data.shape[0]):
    sel_rows = [x for x in all_rows if x != row]
    subs = transformed_data[sel_rows, :] - transformed_data[row, :]
    score = np.mean(subs, axis=0)
    top = score.argsort()[-find_per_row:]
    tops.append(top.tolist())

# flatten list
tops = [item for sublist in tops for item in sublist]

print(len(tops))
print(len(set(tops)))

top_ranks = ranks[:, tops]
print(top_ranks.shape)

for i in range(0, len(top_ranks)):
    print(" ".join([rmse_files[i][:10].ljust(10)] + [("%d" % v).ljust(5) for v in top_ranks[i, :]]))

# """ pack rank to table """
# domain = Domain([ContinuousVariable(alg) for alg in np.array(list(images))[tops]], metas=[StringVariable("Methods")])
# table = Table(domain, top_ranks, metas=[[r] for r in rmse_files])
# table.save("../../processed_data/ranks_top_100_per_method.tab")
#
# domain = Domain([ContinuousVariable(alg) for alg in rmse_files], metas=[StringVariable("Images")])
# table = Table(domain, top_ranks.T, metas=[[r] for r in np.array(list(images))[tops]])
# table.save("../../processed_data/ranks_top_100_per_image.tab")

""" Trial with outliers"""
# implemented looking at this paper about outlier detection with knn
# ftp://ftp10.us.freebsd.org/users/azhang/disc/disc01/cd1/out/papers/sigmod/efficientalgorisrrak.pdf
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(transformed_data.T)

distance_to_neighbours = neigh.kneighbors(transformed_data.T)
max_dist_idx = distance_to_neighbours[0][:, -1].argsort()[-100:]
print(max_dist_idx.shape)

print(len(set(max_dist_idx) - set(tops)))

top_ranks_knn = ranks[:, max_dist_idx]
for i in range(0, len(top_ranks_knn)):
    print(" ".join([rmse_files[i][:10]] + [("%d" % v).ljust(5) for v in top_ranks_knn[i, :]]))

# domain = Domain([ContinuousVariable(alg) for alg in rmse_files], metas=[StringVariable("Images")])
# table = Table(domain, top_ranks_knn.T, metas=[[r] for r in np.array(list(images))[max_dist_idx]])
# table.save("../../processed_data/ranks_top_100_per_image_knn.tab")
#
# domain = Domain([ContinuousVariable(alg) for alg in np.array(list(images))[max_dist_idx]], metas=[StringVariable("Methods")])
# table = Table(domain, top_ranks_knn, metas=[[r] for r in rmse_files])
# table.save("../../processed_data/ranks_top_100_per_method_knn.tab")

# """ SVD """
# U, s, Vh = linalg.svd(ranks, full_matrices=False)
# print(Vh[:, :10])

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