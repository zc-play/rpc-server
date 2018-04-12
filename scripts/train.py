# coding: utf-8
import os
import pickle
import time

import cv2
import face_recognition
import datetime
from config import LFW_MODEL_PATH, VGG_MODEL_PATH, LFW_TRAIN_PATH
from core.face_rec import train, predict_face, draw_labels_and_save
from web.model import Face
from config import LFW_PATH, LFW_TEST_PATH, LFW_TRAIN_PATH
from config import VGG_PATH, VGG_TEST_PATH, VGG_TRAIN_PATH
from core.utils.face_recognition_cli import image_files_in_folder


def train_model(dataset='lfw'):
    print("Training KNN classifier...")
    if dataset == 'lfw':
        model_path = LFW_MODEL_PATH
        train_path = LFW_TRAIN_PATH
    else:
        model_path = VGG_MODEL_PATH
        train_path = VGG_TRAIN_PATH
    if os.path.exists(model_path):
        conform = input('knn classifier already exists. If want to retrain and overwrite it, please press Y: ')
        if conform != 'Y':
            return
    train(train_path, model_save_path=model_path, n_neighbors=3)
    print("Training complete!")


def predict_img(img_path, dataset='lfw'):
    if dataset == 'lfw':
        model_path = LFW_MODEL_PATH
    else:
        model_path = VGG_MODEL_PATH
    with open(model_path, 'rb') as f:
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


def splilt_flw(dataset):
    if dataset == 'lfw':
        img_dir, train_dir, test_dir = LFW_PATH, LFW_TRAIN_PATH, LFW_TEST_PATH
    elif dataset == 'vgg':
        img_dir, train_dir, test_dir = VGG_PATH, VGG_TRAIN_PATH, VGG_TEST_PATH
    else:
        raise Exception()

    count = 0
    for class_dir in os.listdir(img_dir):
        dir_path = os.path.join(img_dir, class_dir)
        if not os.path.isdir(dir_path):
            continue

        if count > 1000:
            break
        count += 1
        image_files = image_files_in_folder(dir_path)
        if len(image_files) < 10:
            continue
        for i, img_path in enumerate(image_files):
            train_path = os.path.join(train_dir, class_dir)
            test_path = os.path.join(test_dir, class_dir)
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            if not os.path.exists(test_path):
                os.mkdir(test_path)

            # 前5个作为训练集体, 剩余的为测试集体
            if i < 5:
                is_train = True
                save_path = os.path.join(train_path, '{}-{}.jpg'.format(class_dir, i))

            elif i < 10:
                is_train = False
                save_path = os.path.join(test_path, '{}-{}.jpg'.format(class_dir, i - 5))
            else:
                continue
            os.system('cp {} {}'.format(img_path, save_path))
            db_img = Face(name=class_dir, path=save_path, is_train=is_train, dataset=dataset)
            db_img.save()


def get_annotation():
    """获取faster rnn annotation"""

    fp = open('/data/algo/data/frnn_annotations.txt', 'w', encoding='utf8')
    f_log = open('/data/algo/data/frnn_annotations_log.txt', 'w', encoding='utf8')
    count = 0
    cache = []
    for class_dir in os.listdir(LFW_PATH):
        dir_path = os.path.join(LFW_PATH, class_dir)
        if not os.path.isdir(dir_path):
            continue

        image_files = image_files_in_folder(dir_path)
        for img_path in image_files:
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            for bbox in face_bounding_boxes:
                # Add face encoding for current image to the training set
                # fp.write('{}, {}, {}, {}, {}, face\n'.format(img_path, bbox[3], bbox[1], bbox[0], bbox[2]))
                cache.append('{}, {}, {}, {}, {}, face\n'.format(img_path, bbox[3], bbox[1], bbox[0], bbox[2]))
            count += 1
            if len(face_bounding_boxes) == 0:
                f_log.write('%s\n' % img_path)
        if len(cache) > 1000:
            fp.writelines(cache)
            cache.clear()
    fp.close()
    f_log.close()


def save_face(image_dir, save_dir):
    count = 0
    for f_dir in os.listdir(image_dir):
        for f_name in os.listdir(os.path.join(image_dir, f_dir)):
            path = os.path.join(image_dir, f_dir, f_name)
            img = cv2.imread(path)
            face_locations = face_recognition.face_locations(img, model='cnn')

            for top, right, bottom, left in face_locations:
                # Draw a box around the face using the Pillow module
                face = img[left:right, top:bottom]
                save_path = os.path.join(save_dir, 'img-{}.jpg'.format(count))
                cols, lens, channels = face.shape
                face = cv2.resize(face, (64, int(64 / cols * cols)))
                cv2.imwrite(save_path, face)
                count += 1


if __name__ == '__main__':
    # predict_img('/data/algo/data/dataset/test/Ahmed_Lopez_0001.jpg')
    # splilt_flw()
    train_model()
    #from core.face_rec import save_face
    #save_face('/data/algo/data/dataset/lfw_train', '/data/algo/data/dataset/lfw_face')
