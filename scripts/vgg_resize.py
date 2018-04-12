# coding: utf-8
import os
import cv2


def resize():
    vgg_path = '/data/dataset/knn/vgg_test'
    vgg_resize_path= '/data/dataset/knn/vgg_test_resize'
    for f_dir in os.listdir(vgg_path):
        f_dir_path = os.path.join(vgg_path, f_dir)
        for f_name in os.listdir(f_dir_path):
            img = cv2.imread(os.path.join(f_dir_path, f_name))
            if img is None:
                continue
            h, w, _ = img.shape
            rh, rw = h, w
            if h > w > 300:
                rh, rw = int(h // (w/300)), 300
                img = cv2.resize(img, (rw, rh))
            elif h <= w and h > 300:
                rh, rw = 300, int(w // (h/300))
                img = cv2.resize(img, (rw, rh))
            path = os.path.join(vgg_resize_path, f_dir)
            os.makedirs(path, exist_ok=True)
            f_path = os.path.join(path, f_name)
            print('{}, convert: ({}, {}) to ({}, {})'.format(path, h, w, rh, rw))
            cv2.imwrite(f_path, img)


resize()
