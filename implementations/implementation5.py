import sys
import os
import time

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Activation, BatchNormalization, UpSampling2D, Lambda
from keras.models import Sequential
from random import shuffle

from keras.utils import HDF5Matrix

from implementations.support_scripts.common import data_to_onehot, H5Choose
from implementations.support_scripts.image_processing import ImageDownloader

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

b_size = 8
dir_name = "../small_dataset"
list_dir = os.listdir(dir_name)
shuffle(list_dir)
list_dir = list_dir
num_classes = 400
n_epochs = 1000

model = Sequential()

# conv1_1
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(256, 256, 1)))
model.add(Activation("relu"))

# conv1_2
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv2_1
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv2_2
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

model.add(Conv2D(400, (1, 1), padding="same"))


# multidimensional softmax
def custom_softmax(x):
    x = K.reshape(x, (b_size * 256 * 256, num_classes))
    x = K.softmax(x)
    x = K.reshape(x, (b_size, 256, 256, num_classes))
    return x

model.add(Activation(custom_softmax))


# sgd = optimizers.SGD(lr=10, momentum=0.0, decay=0, nesterov=False)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,
              loss="mean_squared_error")

model.summary()


save_every_n_epoch = 50
start_from = 100

# start image downloader
id = ImageDownloader("../h5_data", "imp5_")
id.setDaemon(True)  # thread die when main thread die
id.start()

file_picker = H5Choose(dir="../h5_data")

try:
    for epoch in range(n_epochs):
        # Instantiating HDF5Matrix for the training set, which is a slice of the first 150 elements
        file = file_picker.pick_next(id)
        X_train = HDF5Matrix(file, 'grayscale')
        y_train = HDF5Matrix(file, 'ab_hist')

        print("Epoch " + str(epoch) + "/" + str(n_epochs))
        start = time.time()
        for b in range(len(y_train) // b_size):
            i, j = b * b_size, (b+1) * b_size

            a = data_to_onehot(y_train[i:j])
            model.train_on_batch(X_train[i:j], a)
        print("Spent: " + str(time.time() - start))
        if epoch % 5 == 4:
            print(model.evaluate(X_train[:8], data_to_onehot(y_train[:8]), batch_size=8))
        if epoch % 10 == 9:
            model.save_weights("../weights/implementation5-" + str(epoch) + ".h5")

except (KeyboardInterrupt, SystemExit):
    id.stop()
    sys.exit()

id.stop()