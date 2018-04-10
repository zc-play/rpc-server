# coding: utf-8
import math
import os
import pickle

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from models.api import face_detect
from sklearn import neighbors

from config import ALLOWED_EXTENSIONS
from config import LFW_PATH, LFW_TEST_PATH, LFW_TRAIN_PATH
from core.utils.face_recognition_cli import image_files_in_folder
from web.model import Face


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict_face(img_path, knn_clf=None, model_path=None, distance_threshold=0.6, method='dlib'):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :param method: dlib or facenet
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(img_path) or os.path.splitext(img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    if method == 'dlib':
        img = face_recognition.load_image_file(img_path)
    else:
        img = cv2.imread(img_path)
    # face_locations = face_recognition.face_locations(img)
    face_locations = face_detect(img, method=method, format='css')

    # If no faces are found in the image, return an empty result.
    if len(face_locations) == 0:
        return img, []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=3)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    data = []
    for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_locations, are_matches):
        if rec:
            data.append((pred, loc))
        else:
            data.append(('unkown', loc))

    return img, data


def draw_labels_and_save(img, predictions, save_path=None, is_save=True):
    """
    :param img: Image or path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    if not type(img) == np.ndarray:
        img = Image.open(img).convert("RGB")
    else:
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    if is_save:
        img.save(save_path)
    return img


def splilt_flw(train_dir=LFW_PATH):
    for class_dir in os.listdir(train_dir):
        dir_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(dir_path):
            continue

        image_files = image_files_in_folder(dir_path)
        if len(image_files) < 10:
            continue
        for i, img_path in enumerate(image_files):
            train_path = os.path.join(LFW_TRAIN_PATH, class_dir)
            test_path = os.path.join(LFW_TEST_PATH, class_dir)
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            if not os.path.exists(test_path):
                os.mkdir(test_path)

            # 前5个作为训练集体, 剩余的为测试集体
            if i < 5:
                is_train = True
                save_path = os.path.join(train_path, '{}-{}.jpg'.format(class_dir, i))

            else:
                is_train = False
                save_path = os.path.join(test_path, '{}-{}.jpg'.format(class_dir, i - 5))
            os.system('cp {} {}'.format(img_path, save_path))
            db_img = Face(name=class_dir, path=save_path, is_train=is_train)
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


if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict_face(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        draw_labels_and_save(os.path.join("knn_examples/test", image_file), predictions)


print(predict_face)