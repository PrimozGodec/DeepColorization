from keras.losses import categorical_crossentropy
import numpy as np
import keras.backend as K

a = np.array([[[1, 0], [0.7, 0.3]], [[1, 0], [0.7, 0.3]]])

true = np.array([[[1, 0], [0.7, 0.3]], [[1, 0], [0.7, 0.3]]])

a = a[np.newaxis, :, :, :]
true = true[np.newaxis, :, :, :]
print(a.shape)


def custom_crossentropy(y_true, y_pred):

    # Flatten
    nb, h, w, q = y_true.shape
    print(nb, h, w, q)
    y_true = K.reshape(y_true, (nb * h * w, q))
    y_pred = K.reshape(y_pred, (nb * h * w, q))

    cross_ent = K.categorical_crossentropy(y_true, y_pred)
    return K.mean(cross_ent, axis=-1)



def custom_crossentropy1(y_true, y_pred):

    cross_ent = K.categorical_crossentropy(y_true, y_pred)
    return K.mean(cross_ent)


a = K.variable(value=a)
true = K.variable(value=true)

print(K.eval(custom_crossentropy(true, a)))