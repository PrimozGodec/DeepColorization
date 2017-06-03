import sys
import os

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, UpSampling2D, Lambda
from keras.models import Sequential
from random import shuffle

from keras.utils import HDF5Matrix
from implementations.support_scripts import image_processing

from implementations.support_scripts.common import make_prediction_sample, data_to_onehot
from implementations.support_scripts.image_processing import load_images, images_to_l

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

b_size = 16
dir_name = "../small_dataset"
list_dir = os.listdir(dir_name)
shuffle(list_dir)
list_dir = list_dir
num_classes = 400
n_epochs = 1000

model = Sequential()

# conv1_1
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(256, 256, 1)))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv1_2
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv2_1
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv2_2
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv3_1
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv3_2
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv3_3
model.add(Conv2D(256, (3, 3), padding="same", strides=(2, 2)))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv4_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv4_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv4_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv5_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv5_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv5_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv6_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv6_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv6_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv7_1
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv7_2
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv7_3
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# # conv8_1
# model.add(Conv2DTranspose(128, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# # conv8_2
# model.add(Conv2DTranspose(128, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# # conv8_3
# model.add(Conv2DTranspose(64, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# # conv8_4
# model.add(Conv2D(400, (3, 3), padding="same", activation='relu'))

# conv8_1
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv8_2
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv8_3
model.add(Conv2D(400, (1, 1), padding="same"))


# multidimensional softmax
def custom_softmax(x):
    # e = K.exp(x - K.max(x, axis=1, keepdims=True))
    # s = K.sum(e, axis=1, keepdims=True)
    # return e / s
    x = K.reshape(x, (b_size * 64 * 64, num_classes))
    x = K.softmax(x)
    x = K.reshape(x, (b_size, 64, 64, num_classes))
    return x


def resize_image(x):
    return K.resize_images(x, 4, 4, "channels_last")


#
#
# def mean_squared_error(y_true, y_pred):
#     y_true = data_to_onehot(y_true)
#     return K.mean(K.square(y_pred - y_true), axis=-1)



model.add(Activation(custom_softmax))
model.add(Lambda(resize_image))


# sgd = optimizers.SGD(lr=10, momentum=0.0, decay=0, nesterov=False)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,
              loss="mean_squared_error")

model.summary()


save_every_n_epoch = 5
start_from = 100


# Instantiating HDF5Matrix for the training set, which is a slice of the first 150 elements
X_train = HDF5Matrix('../h5_data/test.h5', 'grayscale')
y_train = HDF5Matrix('../h5_data/test.h5', 'ab_hist')

import time

for epoch in range(n_epochs):
    print("Epoch " + str(epoch) + "/" + str(n_epochs))
    start = time.time()
    for b in range(len(y_train) // b_size):
        i, j = b * b_size, (b+1) * b_size
        a = data_to_onehot(y_train)
        model.fit(X_train, a, epochs=1, batch_size=b_size)
    print("Spent:" + str(time.time() - start))