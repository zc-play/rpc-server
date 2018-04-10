# coding: utf-8
# -*- coding: utf-8 -*-
"""
train face-net for 2-classify
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
from keras.layers import TimeDistributed
from .RoiPoolingConv import RoiPoolingConv
from ..face_classifier.config import Config
from ..face_classifier.net import nn_base


def get_weight_path():
    conf = Config()
    return conf.face_model_path


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 16

    return get_output_length(width), get_output_length(height)


def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=True):
    pooling_regions = 4 

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # fc1
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(1024, activation='relu', name='fc1', trainable=trainable))(out)
    out = TimeDistributed(BatchNormalization(trainable=trainable, name='fc1_bn'))(out)
    out = TimeDistributed(Dropout(0.8))(out)
    # fc2
    out = TimeDistributed(Dense(256, activation='relu', name='fc2', trainable=trainable))(out)
    out = TimeDistributed(BatchNormalization(trainable=trainable, name='fc1_bn'))(out)
    out = TimeDistributed(Dropout(0.9))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

