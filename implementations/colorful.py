import sys
import os

import pickle

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Activation, BatchNormalization, UpSampling2D, Lambda
from keras.models import Sequential

from implementations.support_scripts.common import h5_small_vgg_generator_onehot_weights, \
    image_check

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

b_size = 32

model = Sequential()

# conv1_1
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(224, 224, 1)))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv1_2
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv2_1
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv2_2
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv3_1
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv3_2
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv3_3
model.add(Conv2D(256, (3, 3), padding="same", strides=(2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv4_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv4_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv4_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv5_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv5_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv5_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv6_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv6_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv6_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv7_1
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv7_2
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
# conv7_3
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv8_1
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv8_2
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))

# conv8_3
model.add(Conv2D(400, (1, 1), padding="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))


# multidimensional softmax
def reshape_softmax(x):
    shape = K.shape(x)
    x = K.reshape(x, (shape[0] * shape[1] * shape[2], 400))
    x = K.softmax(x)

    # Add a zero column so that x has the same dimension as the target (313 classes + 1 weight)
    xc = K.zeros((b_size * 56 * 56, 1))
    x = K.concatenate([x, xc], axis=-1)

    # Reshape back to (batch_size, h, w, nb_classes + 1) to satisfy keras' shape checks
    x = K.reshape(x, (shape[0], shape[1], shape[2], 400 + 1))
    x = K.resize_images(x, 224 // 56, 224 // 56, "channels_last")

    return x

model.add(Activation(reshape_softmax))


def categorical_crossentropy_color(y_true, y_pred):

    # Flatten
    shape = K.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0] * shape[1] * shape[2], shape[3]))
    y_true = K.reshape(y_true, (shape[0] * shape[1] * shape[2], shape[3]))

    weights = y_true[:, 400:]  # extract weight from y_true
    weights = K.concatenate([weights] * 400, axis=1)
    y_true = y_true[:, :-1]  # remove last column
    y_pred = y_pred[:, :-1]  # remove last column

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    cross_ent = K.mean(cross_ent, axis=-1)

    return cross_ent

opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss=categorical_crossentropy_color)

model.summary()
exit()


start_from = 0
save_every_n_epoch = 5
n_epochs = 10000
# model.load_weights("../weights/vgg1-0.h5")

g = h5_small_vgg_generator_onehot_weights(b_size, "../h5_data_224", None)
gval = h5_small_vgg_generator_onehot_weights(b_size, "../h5_data_224_validate", None)


for i in range(start_from // save_every_n_epoch, n_epochs // save_every_n_epoch):
    print("START", i * save_every_n_epoch, "/", n_epochs)
    history = model.fit_generator(g, steps_per_epoch=60000/b_size, epochs=save_every_n_epoch,
                                  validation_data=gval, validation_steps=(1024//b_size))
    model.save_weights("../weights/colorful-" + str(i * save_every_n_epoch) + ".h5")

    # save sample images
    image_check(model, 40, "colorful-" + str(i * save_every_n_epoch) + "-")

    # save history
    output = open('../history/colorful-{:0=4d}.pkl'.format(i * save_every_n_epoch), 'wb')
    pickle.dump(history.history, output)
    output.close()
