import matplotlib.pyplot as plt
import numpy as np

"""
This script show data chart analysis for training
"""

# select implementation
import os
import pickle
from matplotlib.ticker import MaxNLocator

implementation = "hist05"

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

ax = plt.figure().gca()
plt.plot(loss)
plt.plot(val_loss)

print(len(loss))
print(loss)
print(val_loss)
print(np.argmin(val_loss))

# plt.title('model loss')
plt.ylabel('Vrednost cenilne funkcije')
plt.xlabel('Korak')
plt.legend(['Učna množica', 'Validacijska množica'], loc='upper right')
plt.ylim([2.25, 2.55])


ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# plt.show()
plt.savefig("../../hist_graphs/" + implementation + ".jpg")