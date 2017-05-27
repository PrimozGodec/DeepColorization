from keras.models import Sequential
from keras.layers import Activation, Conv2D, Conv2DTranspose
from keras import backend as K

def custom_softmax(x):
    e = K.exp(x - K.max(x, axis=0, keepdims=True))
    s = K.sum(e, axis=0, keepdims=True)
    return e / s



model = Sequential()

# conv1_1
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256)))
# conv1_2
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))

# conv2_1
model.add(Conv2D(128, (3, 3), activation='relu'))
# conv2_2
model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu'))

# conv3_1
model.add(Conv2D(256, (3, 3), activation='relu'))
# conv3_2
model.add(Conv2D(256, (3, 3), activation='relu'))
# conv3_3
model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu'))

# conv4_1
model.add(Conv2D(512, (3, 3), activation='relu'))
# conv4_2
model.add(Conv2D(512, (3, 3), activation='relu'))
# conv4_3
model.add(Conv2D(512, (3, 3), activation='relu'))

# conv5_1
model.add(Conv2D(512, (3, 3), activation='relu'))
# conv5_2
model.add(Conv2D(512, (3, 3), activation='relu'))
# conv5_3
model.add(Conv2D(512, (3, 3), activation='relu'))

# conv6_1
model.add(Conv2D(512, (3, 3), activation='relu'))
# conv6_2
model.add(Conv2D(512, (3, 3), activation='relu'))
# conv6_3
model.add(Conv2D(512, (3, 3), activation='relu'))

# conv7_1
model.add(Conv2D(256, (3, 3), activation='relu'))
# conv7_2
model.add(Conv2D(256, (3, 3), activation='relu'))
# conv7_3
model.add(Conv2D(256, (3, 3), activation='relu'))

# conv8_1
model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu'))
# conv8_2
model.add(Conv2D(128, (3, 3), activation='relu'))
# conv8_3
model.add(Conv2D(64, (3, 3), activation='relu'))
# conv8_4
model.add(Conv2D(2, (3, 3), activation='relu'))

# multidimensional softmax
# todo: try the way to use default function - with axis
model.add(Activation(custom_softmax))


model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu'))
