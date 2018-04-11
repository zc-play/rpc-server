# coding: utf-8
import os
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# data path
ROOT_PATH = "/data"
LFW_MODEL_PATH = os.path.join(ROOT_PATH, 'models', 'lfw_knn_face_rec_model.clf')

LFW_TRAIN_PATH = os.path.join(ROOT_PATH, 'dataset', 'knn', 'lfw_train')
LFW_TEST_PATH = os.path.join(ROOT_PATH, 'dataset', 'knn', 'lfw_test')
LFW_PATH = os.path.join(ROOT_PATH, 'dataset', 'knn', 'lfw')

os.makedirs(LFW_TRAIN_PATH, exist_ok=True)
os.makedirs(LFW_TEST_PATH, exist_ok=True)

VGG_TRAIN_PATH = os.path.join(ROOT_PATH, 'dataset', 'knn', 'vgg_train')
VGG_TEST_PATH = os.path.join(ROOT_PATH, 'dataset', 'knn', 'vgg_test')
VGG_PATH = os.path.join(ROOT_PATH, 'dataset', 'vgg_face', 'images')
VGG_MODEL_PATH = os.path.join(ROOT_PATH, 'models', 'vgg_knn_face_rec_model.clf')

os.makedirs(VGG_TRAIN_PATH, exist_ok=True)
os.makedirs(VGG_TEST_PATH, exist_ok=True)
