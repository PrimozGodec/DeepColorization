import json
import numpy as np


with open("../../../admin.json") as data_file:
    data = json.load(data_file)

methods = []
for item in data:
    if item["fields"]["dataset1"] not in methods:
        methods.append(item["fields"]["dataset1"])
    if item["fields"]["dataset2"] not in methods:
        methods.append(item["fields"]["dataset2"])

print(methods)

counts = np.zeros((len(methods), len(methods)))

for item in data:
    win = item["fields"]["dataset1"] if item["fields"]["estimated_original"] else item["fields"]["dataset2"]
    loose = item["fields"]["dataset2"] if item["fields"]["estimated_original"] else item["fields"]["dataset1"]

    i = methods.index(win)
    j = methods.index(loose)

    counts[i, j] += 1

print(counts[2:, 2:])
