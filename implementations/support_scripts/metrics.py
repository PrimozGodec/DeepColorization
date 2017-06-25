import keras.backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.mean(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)), axis=[-2, -1])


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3])

