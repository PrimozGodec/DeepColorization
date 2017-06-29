import numpy as np
import os
import pickle
from Orange.projection import CUR
from Orange.data import Table

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
print(data)
cur = CUR(rank=7, max_error=2, compute_U=False)
cur_model = cur(data)

transformed_data = cur_model(data, axis=1)
a = np.argmax(transformed_data, axis=1)
print(transformed_data)

print(a)