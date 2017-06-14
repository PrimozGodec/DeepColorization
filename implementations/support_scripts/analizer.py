import matplotlib.pyplot as plt

"""
This script show data chart analysis for training
"""

# select implementation
import os
import pickle

implementation = "imp7d"

# list all history files
files = sorted([x for x in os.listdir("../../history") if x.startswith(implementation)])

loss = []
val_loss = []
for file in files:
    with open("../../history/" + file, "rb") as f:
        data = pickle.load(f)
        loss += data["loss"]
        val_loss += data["val_loss"]

plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()