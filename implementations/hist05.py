import pickle
import sys
import os

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])

from implementations.support_scripts.image_tester import image_error_small_hist
from implementations.support_scripts.metrics import root_mean_squared_error, mean_squared_error

from implementations.support_scripts.common import whole_image_check, h5_small_vgg_generator, \
    h5_small_vgg_generator_onehot, whole_image_check_hist, h5_small_vgg_generator_onehot_neigh, \
    h5_small_vgg_generator_onehot_weight_hist04
from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K, Input
from keras import optimizers
from keras.layers import Conv2D, UpSampling2D, Lambda, Dense, Merge, merge, concatenate, Activation

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

b_size = 32

num_classes = 400
input_shape = (32, 32, 1)

# main network
main_input = Input(shape=input_shape, name='image_part_input')

x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(main_input)
x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
main_output = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

# VGG
vgg16 = VGG16(weights="imagenet", include_top=True)
vgg_output = Dense(256, activation='softmax', name='predictions')(vgg16.layers[-2].output)

def repeat_output(input):
    shape = K.shape(x)
    return K.reshape(K.repeat(input, 4 * 4), (shape[0], 4, 4, 256))

vgg_output = Lambda(repeat_output)(vgg_output)

# freeze vgg16
for layer in vgg16.layers:
    layer.trainable = False

# concatenated net
merged = concatenate([vgg_output, main_output], axis=3)

last = Conv2D(256, (3, 3), padding="same")(merged)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)
last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)
last = Conv2D(400, (3, 3), padding="same", activation="relu")(last)


def resize_image(x):
    return K.resize_images(x, 2, 2, "channels_last")


# multidimensional softmax
def custom_softmax(x):
    sh = K.shape(x)
    x = K.reshape(x, (sh[0] * sh[1] * sh[2], num_classes))
    x = K.softmax(x)
    xc = K.zeros((b_size * 16 * 16, 1))
    x = K.concatenate([x, xc], axis=-1)

    x = K.reshape(x, (sh[0], sh[1], sh[2], num_classes + 1))
    return x


last = Activation(custom_softmax)(last)
last = Lambda(resize_image)(last)


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


model = Model(inputs=[main_input, vgg16.input], output=last)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss=categorical_crossentropy_color, metrics=[root_mean_squared_error, mean_squared_error])

model.summary()

start_from = 0
save_every_n_epoch = 1
n_epochs = 150

print("weights loaded")
# model.load_weights("../weights/hist05-14.h5")

# start image downloader
# ip = ImagePacker("../small_dataset", "../h5_data",  "imp7d-", num_images=1024, num_files=None)
# ip.start()
ip = None

g = h5_small_vgg_generator_onehot_weight_hist04(b_size, "../data/h5_small_train", ip)
gval = h5_small_vgg_generator_onehot_weight_hist04(b_size, "../data/h5_small_validation", None)


for i in range(start_from // save_every_n_epoch, n_epochs // save_every_n_epoch):
    print("START", i * save_every_n_epoch, "/", n_epochs)
    history = model.fit_generator(g, steps_per_epoch=50000//b_size, epochs=save_every_n_epoch,
                                  validation_data=gval, validation_steps=(10000//b_size))
    model.save_weights("../weights/hist05-" + str(i * save_every_n_epoch) + ".h5")

    # save sample images
    whole_image_check_hist(model, 80, "hist05-" + str(i * save_every_n_epoch) + "-")

    # save history
    output = open('../history/hist05-{:0=4d}.pkl'.format(i * save_every_n_epoch), 'wb')
    pickle.dump(history.history, output)
#     output.close()

# image_error_small_hist(model, "hist05-test-100")