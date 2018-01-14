# -*- coding:utf-8 -*-
"""

A reproduce of ResNet with Identity Mappings
author: Tianz

"""

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input,add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def residual_block(x, nb_filters,strides=(1,1),
               dropout_rate=0., weight_decay=1E-4):
    x = Conv2D(nb_filters, (3,3),
                        kernel_initializer='he_normal',
                        padding="same",
                        strides=strides,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filters, (3, 3),
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    return x


def create_ResNet(nb_classes, img_dim, nb_blocks=[4,4,4],k=1,
                      weight_decay=1E-4,droprate=0.):
    """
    :param nb_classes: the number of your dataset classes,
    for cifar-10, nb_classes should be 10
    :param img_dim: the input shape of the model input
    :param nb_blocks: the number of blocks in each stage
    :param k: the widen fatcor, k=1 indicates that the model
    is original ResNet, when k>1 the model is a wide ResNet
    :param weight_decay: weight decay for L2 regularization
    :param droprate: the dropout between two convolutons of
    each block and the default drop rate is set to 0.0
    :return: ResNet model or WRN model
    """
    model_input = Input(shape=img_dim)
    stack = [16*k, 32*k, 64*k]
    nb_filter = 16
    # Initial convolution
    y = Conv2D(nb_filter, (3, 3),
                      kernel_initializer="he_normal",
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(model_input)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    x = Activation('relu')(x)

    # stage 1
    x = residual_block(x,stack[0],dropout_rate=droprate)
    if stack[0] != 16:
        y = Conv2D(stack[0], (1, 1),
                          kernel_initializer="he_normal",
                          padding="same",
                          use_bias=False,
                          kernel_regularizer=l2(weight_decay))(y)
    y = add([x,y])
    for j in range(nb_blocks[0]-1):
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
        x = Activation('relu')(x)
        x = residual_block(x, stack[0],dropout_rate=droprate)

        y = add([x,y])
        if droprate:
            y = Dropout(droprate)(y)

    # stage 2
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    y = Activation('relu')(x)

    x = residual_block(y,stack[1],strides=(2,2),dropout_rate=droprate)
    y = Conv2D(stack[1], (1, 1),strides=(2,2),
               kernel_initializer="he_normal",
               padding="valid",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)
    y = add([x, y])

    for j in range(nb_blocks[1]-1):
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
        x = Activation('relu')(x)
        x = residual_block(x, stack[1],dropout_rate=droprate)

        y = add([x, y])

    # stage 3
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    y = Activation('relu')(x)

    x = residual_block(y, stack[2],strides=(2,2),dropout_rate=droprate)
    y = Conv2D(stack[2], (1, 1),strides=(2,2),
               kernel_initializer="he_normal",
               padding="valid",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)
    y = add([x, y])

    for j in range(nb_blocks[2]-1):
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
        x = Activation('relu')(x)
        x = residual_block(x, stack[2],dropout_rate=droprate)

        y = add([x, y])

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    model = Model(input=[model_input], output=[x])

    return model