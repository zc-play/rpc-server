# coding: utf-8
import os
from ..keras_frcnn.config import DATA_ROOT_PATH, MODEL_ROOT_PATH, LOG_ROOT_PATH


class Config(object):

    def __init__(self):
        self.data_root_path = os.path.join(DATA_ROOT_PATH, 'face_classify')
        self.train_path = os.path.join(self.data_root_path, 'train')
        self.dev_path = os.path.join(self.data_root_path, 'dev')
        self.test_path = os.path.join(self.data_root_path, 'test')
        self.train_log = os.path.join(LOG_ROOT_PATH, 'face-classifier')
        self.keras_vgg16_path = os.path.join(MODEL_ROOT_PATH, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        self.face_model_path = os.path.join(MODEL_ROOT_PATH, 'face-classify', 'face-classifier.h5')
