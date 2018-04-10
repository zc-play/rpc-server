# coding: utf-8
import os
from ..keras_frcnn.config import DATA_ROOT_PATH, MODEL_ROOT_PATH, LOG_ROOT_PATH
from ..face_classifier import config


class Config(object):

    def __init__(self):
        self.data_root_path = os.path.join(DATA_ROOT_PATH, 'face-recognition')
        self.train_path = os.path.join(self.data_root_path, 'train')
        self.dev_path = os.path.join(self.data_root_path, 'dev')
        self.train_log = os.path.join(LOG_ROOT_PATH, 'face-recognition')

        # output tensor numbers of the model
        self.class_size = 2622   # len(os.listdir(self.train_path))                      #
        # pre_train form face classifier model
        self.pre_train_model_path = config.Config().face_model_path
        self.model_path = os.path.join(MODEL_ROOT_PATH, 'face-recognition', 'face-recognition.h5')
