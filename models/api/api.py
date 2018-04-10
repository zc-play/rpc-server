# coding: utf-8
import time
import face_recognition
from .utils import face_detector


def face_rec():
    pass


def face_detect(img, method='face-net', format='cv'):
    res = []
    t1_record = time.time()
    if method == 'dlib':
        locations = face_recognition.face_locations(img, model='cnn')
        # convert (top, right, bottom, left) to (x1, y1, x2, y2)
        if format == 'cv':
            for top, right, bottom, left in locations:
                res.append([left, top, right, bottom])
        else:
            res = locations
    elif method == 'face-net':
        locations = face_detector.detect_face(img)
        if format == 'css':
            for left, top, right, bottom in locations:
                res.append([top, right, bottom, left])
        else:
            res = locations
    else:
        raise Exception('no support this method')
    print('face_detect, method: {}, elapsed time: {}, res: {}'.format(method, time.time() - t1_record, locations))
    return res


def face_encoding(face, method='face-net'):
    if method == 'dlib':
        encoding = face_recognition.face_encodings(face)
    elif method == 'face-net':
        encoding = None     # todo
    else:
        raise Exception('no support this method')
    return encoding




