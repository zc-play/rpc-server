# coding: utf-8
import os
import pickle
import time

import datetime
from config import MODEL_PATH, LFW_TRAIN_PATH
from core.face_rec import train, predict_face, draw_labels_and_save


def train_model():
    print("Training KNN classifier...")
    if os.path.exists(MODEL_PATH):
        conform = input('knn classifier already exists. If want to retrain and overwrite it, please press Y: ')
        if conform != 'Y':
            return
    train(LFW_TRAIN_PATH, model_save_path=MODEL_PATH, n_neighbors=3)
    print("Training complete!")


def predict_img(img_path):
    with open(MODEL_PATH, 'rb') as f:
        knn_clf = pickle.load(f)
    t_record1 = time.time()
    predictions = predict_face(img_path, knn_clf=knn_clf, distance_threshold=0.4)
    # Print results on the console
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))
    date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d.%H%M%S')
    save_path = 'res_{}.jpg'.format(date_str)
    t_record2 = time.time()
    draw_labels_and_save(os.path.join("tmp", img_path), predictions, save_path)
    print('predict time: {}s'.format(t_record2 - t_record1))


if __name__ == '__main__':
    # predict_img('/data/algo/data/dataset/test/Ahmed_Lopez_0001.jpg')
    # splilt_flw()
    train_model()
    #from core.face_rec import save_face
    #save_face('/data/algo/data/dataset/lfw_train', '/data/algo/data/dataset/lfw_face')
