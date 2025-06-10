import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model


def conv2d(x, filters, kernel, strides, include_bn_elu=True):
    x = layers.Conv2D(
        filters=filters, kernel_size=kernel, strides=strides, padding="same"
    )(x)
    if include_bn_elu:
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
        x = tf.keras.layers.ELU()(x)
    return x


def get_model(num_freq, num_frames, num_channels):
    inputs = Input(shape=(num_freq, num_frames, num_channels))

    x = conv2d(inputs, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = skip1 = conv2d(x, 64, kernel=(5, 5), strides=(2, 2), include_bn_elu=False)

    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
    x = tf.keras.layers.ELU()(x)
    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = skip2 = conv2d(x, 64, kernel=(5, 5), strides=(2, 2), include_bn_elu=False)

    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
    x = tf.keras.layers.ELU()(x)
    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = skip3 = conv2d(x, 64, kernel=(5, 5), strides=(2, 2), include_bn_elu=False)

    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
    x = tf.keras.layers.ELU()(x)
    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = skip4 = conv2d(x, 64, kernel=(5, 5), strides=(2, 2), include_bn_elu=False)

    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
    x = tf.keras.layers.ELU()(x)
    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = conv2d(x, 64, kernel=(5, 5), strides=(2, 2), include_bn_elu=True)

    x = conv2d(x, 64, kernel=(3, 7), strides=(1, 1), include_bn_elu=True)
    x = conv2d(x, 64, kernel=(3, 7), strides=(1, 1), include_bn_elu=True)
    x = conv2d(x, 64, kernel=(3, 7), strides=(1, 1), include_bn_elu=True)

    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding="same")(x)

    x += skip4
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
    x = tf.keras.layers.ELU()(x)
    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding="same")(x)

    x += skip3
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
    x = tf.keras.layers.ELU()(x)
    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding="same")(x)

    x += skip2
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
    x = tf.keras.layers.ELU()(x)
    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding="same")(x)

    x += skip1
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
    x = tf.keras.layers.ELU()(x)
    x = conv2d(x, 64, kernel=(3, 3), strides=(1, 1), include_bn_elu=True)
    x = layers.Conv2DTranspose(3, (6, 6), strides=(2, 2), padding="same")(x)

    return Model(inputs, x)
