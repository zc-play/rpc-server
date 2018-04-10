# coding: utf-8
import os
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# data path
ROOT_PATH = "/data"
MODEL_PATH = os.path.join(ROOT_PATH, 'models', 'knn_face_rec_model.clf')

LFW_TRAIN_PATH = os.path.join(ROOT_PATH, 'dataset', 'lfw_train')
LFW_TEST_PATH = os.path.join(ROOT_PATH, 'dataset', 'lfw_test')
LFW_PATH = os.path.join(ROOT_PATH, 'dataset', 'lfw')

os.makedirs(LFW_TRAIN_PATH, exist_ok=True)
os.makedirs(LFW_TEST_PATH, exist_ok=True)
