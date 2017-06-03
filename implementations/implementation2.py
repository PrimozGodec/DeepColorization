import os

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, UpSampling2D, Lambda
from keras.models import Sequential
from random import shuffle


from support_scripts import image_processing

from implementations.support_scripts.common import make_prediction_sample

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# cross entropy
def custom_crossentrophy(y_true, y_pred):

    # Flatten
    tf_session = K.get_session()
    [nb, h, w, q] = K.shape(y_pred).eval(session=tf_session)

    y_true = K.reshape(y_true, (nb * h * w, q))
    y_pred = K.reshape(y_pred, (nb * h * w, q))

    cross_ent = K.categorical_crossentropy(y_true, y_pred)
    return K.mean(cross_ent, axis=-1)


b_size = 2
images_dir_name = "../small_dataset"
list_dir = os.listdir(images_dir_name)
shuffle(list_dir)
list_dir = list_dir
num_classes = 400
n_epochs = 100

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
    x = K.resize_images(x, 4, 4, "channels_last")
    return x


model.add(Activation(custom_softmax))
model.add(Lambda(resize_image))


# sgd = optimizers.SGD(lr=10, momentum=0.0, decay=0, nesterov=False)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,
              loss=custom_crossentrophy)

model.summary()
model.load_weights("../weights/implementation2-95.h5")

save_every_n_epoch = 5

# for i in range(n_epochs // save_every_n_epoch):
#     model.fit_generator(image_processing.image_generator_hist(list_dir, images_dir_name, b_size),
#                      steps_per_epoch=len(list_dir)//b_size, epochs=save_every_n_epoch)
#     model.save_weights("../weights/implementation2-" + str(i * save_every_n_epoch) + ".h5")
#
#     # make validation
#     loss = model.evaluate_generator(image_processing.image_generator_hist(None, "../test_set", b_size), 2)
#     print("Validation loss:", loss)



# g = image_processing.image_generator_hist(list_dir, 1)
# i = next(g)
#
# model.predict(image_processing.load_images(i[0]))


# model.load_weights('implementation1_1.h5')
#
# prediction = model.predict(images_to_l(load_images(list_dir[0])))
#
# with open('pred.pickle', 'wb') as handle:
#     pickle.dump(prediction, handle)

make_prediction_sample(model, b_size, "im2-200-")