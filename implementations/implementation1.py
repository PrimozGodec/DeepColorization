import os

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, UpSampling2D, Lambda
from keras.models import Sequential
from random import shuffle

from support_scripts import image_processing

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# cross entropy
# def categorical_crossentropy_color(y_true, y_pred):
#
#     # Flatten  # todo: make weight
#     n, h, w, q = y_true.shape
#     y_true = K.reshape(y_true, (n * h * w, q))
#     y_pred = K.reshape(y_pred, (n * h * w, q))
#
#     weights = y_true[:, 313:]  # extract weight from y_true
#     weights = K.concatenate([weights] * 313, axis=1)
#     y_true = y_true[:, :-1]  # remove last column
#     y_pred = y_pred[:, :-1]  # remove last column
#
#     # multiply y_true by weights
#     y_true = y_true * weights
#
#     cross_ent = K.categorical_crossentropy(y_pred, y_true)
#     cross_ent = K.mean(cross_ent, axis=-1)
#
#     return cross_ent


b_size = 2
list_dir = os.listdir("../small_dataset")
shuffle(list_dir)
list_dir = list_dir[:20]
num_classes = 400

model = Sequential()

# conv1_1
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(256, 256, 1)))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv1_2
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv2_1
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv2_2
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv3_1
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv3_2
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv3_3
model.add(Conv2D(256, (3, 3), padding="same", strides=(2, 2)))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv4_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv4_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv4_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv5_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv5_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv5_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv6_1
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv6_2
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv6_3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv7_1
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv7_2
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))
# conv7_3
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
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
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
model.add(Activation("relu"))

# conv8_2
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(BatchNormalization(axis=1))  # todo: check if really axis 1 since data has last axis for chanel
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
    x = K.resize_images(x, 4, 4, "channels_last")
    return x


model.add(Activation(custom_softmax))
model.add(Lambda(resize_image))


# sgd = optimizers.SGD(lr=10, momentum=0.0, decay=0, nesterov=False)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mse'])

model.summary()


model.fit_generator(image_processing.image_generator_hist(list_dir, b_size),
                    steps_per_epoch=len(list_dir)//b_size, epochs=1)
model.save_weights("implementation1.h5")


# g = image_processing.image_generator_hist(list_dir, 1)
# i = next(g)
#
# model.predict(image_processing.load_images(i[0]))


model.save_weights('implementation1_1.h5')
# model.load_weights('implementation1_1.h5')
#
# prediction = model.predict(images_to_l(load_images(list_dir[0])))
#
# with open('pred.pickle', 'wb') as handle:
#     pickle.dump(prediction, handle)