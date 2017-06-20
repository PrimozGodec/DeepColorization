import matplotlib.pyplot as plt
import numpy as np

"""
This script show data chart analysis for training
"""

# select implementation
import os
import pickle

implementation = "imp7d-hist"

# list all history files
files = sorted([x for x in os.listdir("../../history")
                if x.startswith(implementation) and len(x[len(implementation) + 1:].split("-")) == 1
                and x[len(implementation)] == "-"])

loss = []
val_loss = []
for file in files:
    print(file)
    with open("../../history/" + file, "rb") as f:
        data = pickle.load(f)
        loss += data["loss"]
        val_loss += data["val_loss"]

plt.plot(loss)
plt.plot(val_loss)

print(len(loss))
# print(np.mean(loss[80:135]))
# print(np.mean(val_loss[80:135]))

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.ylim([160, 280])

plt.savefig("../../hist_graphs/" + implementation + ".jpg")