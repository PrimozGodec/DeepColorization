import json
from collections import Counter

import numpy as np

# open the file with results
with open("../../../responses-28-07-2017-1.json") as data_file:
    data = json.load(data_file)

# collect methods
methods = []
for item in data:
    if item["fields"]["dataset1"] not in methods:
        methods.append(item["fields"]["dataset1"])
    if item["fields"]["dataset2"] not in methods:
        methods.append(item["fields"]["dataset2"])

print(methods)

# collect wind and looses
counts = np.zeros((len(methods), len(methods)))

for item in data:
    win = item["fields"]["dataset1"] if item["fields"]["estimated_original"] else item["fields"]["dataset2"]
    loose = item["fields"]["dataset2"] if item["fields"]["estimated_original"] else item["fields"]["dataset1"]

    i = methods.index(win)
    j = methods.index(loose)

    counts[i, j] += 1

wins = np.sum(counts[2:, 2:], axis=1)

print(counts[2:, 2:])
print(np.sum(counts[2:, 2:], axis=1) + np.sum(counts[2:, 2:], axis=0))
print(np.sum(counts[2:, 2:], axis=1))

# counts without tests
counts_real = counts[2:, 2:]

# check for duplicates
# dict = {}
# for item in data:
#
#     fields = item["fields"]
#     if fields["user_id"] not in dict:
#         dict[fields["user_id"]] = []
#     dict[fields["user_id"]].append(fields["image"])
#
# counts = {}
# for k, v in dict.items():
#     counts[k] = Counter(v)
#     for c, cou in counts[k].items():
#         if cou > 1 and not c.startswith("test"):
#             pass
#             # print(k)

# print(counts[310])

"""
Bradley - Terry
"""
# stop log likelihood diff
ll_stop = 1e-12

# log likelihood
def l(y):
    ll = 0
    for i in range(len(y)):
        for j in range(len(y)):
            ll += counts_real[i, j] * np.log(y[i]) - counts_real[i, j] * np.log(y[i] + y[j])
    return ll

# init y
y = np.ones(len(methods) - 2) / (len(methods) - 2)
prew_l = 0

# every step
for n in range(100):
    # every parameter
    y_ = np.copy(y)
    for i in range(len(y)):
        W = wins[i]
        sum_ = 0
        for j in range(len(y)):
            if i == j:
                continue
            N = counts_real[i, j] + counts_real[j, i]
            sum_ += N / (y[i] + y[j])
        y_[i] = W * (sum_ ** -1)

    # nomralize
    sum_ = np.sum(y_)
    for i in range(len(y)):
        y[i] = y_[i] / sum_
    ll = l(y)
    if abs(ll- prew_l) < ll_stop:
        break
    prew_l = ll
    print(n, ll)

"""
win probabilities
"""
win_matrix = np.zeros(counts_real.shape)
for i in range(len(y)):
    for j in range(len(y)):
        win_matrix[i, j] = y[i] / (y[i] + y[j])

for i in range(win_matrix.shape[0]):
    print(("  ").join(["{0:.2f}".format(x) for x in win_matrix[i, :]]))

"""
Some statistic
"""
# num of users
users = [x["fields"]["user_id"] for x in data if not x["fields"]["dataset1"].startswith("test")]
num_users = len(set(users))
print("St. up.:", num_users)

# average responses per user
count = Counter(users).values()
count_count = Counter(count)
print("Avg. resp. p. user", sum(count) / len(count))
print("More than 21:", sum(value for key, value in count_count.items() if key > 21))
print("21 responses:", sum(value for key, value in count_count.items() if key == 21))
print("Less than 21:", sum(value for key, value in count_count.items() if key < 21))
