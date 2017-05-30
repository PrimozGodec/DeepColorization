import os

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Conv2DTranspose, Activation
from keras.models import Sequential
from random import shuffle

from support_scripts import image_processing

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def custom_softmax(x):
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


b_size = 2
list_dir = os.listdir("../small_dataset")
shuffle(list_dir)
list_dir = list_dir[:20]

model = Sequential()

# conv1_1
model.add(Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(256, 256, 1)))
# conv1_2
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'))

# conv2_1
model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
# conv2_2
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu'))

# conv3_1
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# conv3_2
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# conv3_3
model.add(Conv2D(256, (3, 3), padding="same", strides=(2, 2), activation='relu'))

# conv4_1
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv4_2
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv4_3
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))

# conv5_1
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv5_2
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv5_3
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))

# conv6_1
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv6_2
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))
# conv6_3
model.add(Conv2D(512, (3, 3), padding="same", activation='relu'))

# conv7_1
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# conv7_2
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
# conv7_3
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))

# conv8_1
model.add(Conv2DTranspose(128, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# conv8_2
model.add(Conv2DTranspose(128, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# conv8_3
model.add(Conv2DTranspose(64, (3, 3), padding="same",  strides=(2, 2), activation='relu'))
# conv8_4
model.add(Conv2D(400, (3, 3), padding="same", activation='relu'))

# multidimensional softmax
# todo: try the way to use default function - with axis
model.add(Activation(custom_softmax))


sgd = optimizers.SGD(lr=1, momentum=0.0, decay=0, nesterov=False)
model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=['mse'])

model.summary()

model.fit_generator(image_processing.image_generator_hist(list_dir, b_size),
                    steps_per_epoch=len(list_dir)//b_size, epochs=50)

model.save_weights('implementation1_1.h5')
# model.load_weights('implementation1_1.h5')
#
# prediction = model.predict(images_to_l(load_images(list_dir[0])))
#
# with open('pred.pickle', 'wb') as handle:
#     pickle.dump(prediction, handle)