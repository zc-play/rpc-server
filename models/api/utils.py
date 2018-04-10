# coding: utf-8
import os
import cv2
import time
import pickle
import logging
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras import backend as K
from ..keras_frcnn import face_net as nn
from ..keras_frcnn import roi_helpers
from ..keras_frcnn.config import Config as ConfigFrcnn


class FaceDetect(object):

    def __init__(self, config_path='./config.pickle', model_path=None, num_rois=32):
        self.model_path = model_path
        self.config_path = config_path
        self.num_rois = num_rois
        self.bbox_threshold = 0.5           # Probability threshold of predicted results of bundling box

        # load config
        # with open(self.config_path, 'rb') as fb:
        #    self.cfg = pickle.load(fb)
        self.cfg = ConfigFrcnn()
        self.cfg.class_mapping = {'face': 0, 'bg': 1}
        class_mapping = self.cfg.class_mapping
        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)
        self.class_mapping = {v: k for k, v in class_mapping.items()}

        # model
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, 512)
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        shared_layers = nn.nn_base(img_input, trainable=True)
        num_anchors = len(self.cfg.anchor_box_scales) * len(self.cfg.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)
        classifier = nn.classifier(feature_map_input, roi_input, self.cfg.num_rois, nb_classes=len(class_mapping), trainable=True)

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier = Model([feature_map_input, roi_input], classifier)

        # load weight
        if self.model_path is None:
            self.model_path = self.cfg.model_path
        print('\n\n', self.cfg)
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True)
        # compile
        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')
        self.graph = tf.get_default_graph()

    def detect_face(self, img):
        """
         detect face on the img
        :param img: numpy.ndarray of img or img path
        :return: list of bundling boxes
        """
        # for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        if isinstance(img, str) and os.path.isfile(img) and \
                img.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise Exception('img format error')

        st = time.time()

        X, ratio = self.format_img(img)

        X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        with self.graph.as_default():
            [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, self.cfg, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        # Number of ROIs per iteration
        for jk in range(R.shape[0] // self.cfg.num_rois + 1):
            ROIs = np.expand_dims(R[self.cfg.num_rois * jk:self.cfg.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.cfg.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.cfg.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            with self.graph.as_default():
                [P_cls, P_regr] = self.model_classifier.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                # filter
                if np.max(P_cls[0, ii, :]) < self.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= self.cfg.classifier_regr_std[0]
                    ty /= self.cfg.classifier_regr_std[1]
                    tw /= self.cfg.classifier_regr_std[2]
                    th /= self.cfg.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [self.cfg.rpn_stride * x, self.cfg.rpn_stride * y, self.cfg.rpn_stride * (x + w), self.cfg.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        locations = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            try:
                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.4)
            except Exception as e:
                logging.warning('no_max_suppression_fast error: {}'.format(e))
                continue
            for jk in range(new_boxes.shape[0]):
                x1, y1, x2, y2 = new_boxes[jk, :]
                locations.append(self.get_real_coordinates(ratio, x1, y1, x2, y2))

        len_all_locations = len(bboxes['face']) if 'face' in bboxes else 0
        logging.info('Detect the image, elapsed time = {}, bboxes: {}, '
                     'locations： {}'.format(time.time() - st, len_all_locations, len(locations)))
        return locations

    def format_img_size(self, img):
        """ formats the image size based on config """
        img_min_side = float(self.cfg.im_size)
        (height, width, _) = img.shape

        if width <= height:
            ratio = img_min_side / width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side / height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    def format_img_channels(self, img):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self.cfg.img_channel_mean[0]
        img[:, :, 1] -= self.cfg.img_channel_mean[1]
        img[:, :, 2] -= self.cfg.img_channel_mean[2]
        img /= self.cfg.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(self, img):
        """ formats an image for model prediction based on config """
        img, ratio = self.format_img_size(img)
        img = self.format_img_channels(img)
        return img, ratio

    @staticmethod
    def get_real_coordinates(ratio, x1, y1, x2, y2):
        """Method to transform the coordinates of the bounding box to its original size"""
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return real_x1, real_y1, real_x2, real_y2


def draw_flag(img,  rectangles, texts=None):
    for idx, (x1, y1, x2, y2) in enumerate(rectangles):
        cv2.rectangle(img, (x1, y1), (x2, y2), np.random.randint(0, 255, 3), 2)
        if texts is not None:
            text_label = texts[idx]
            ret_val, base_line = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            text_org = x1, y1 - 0

            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line - 5),
                          (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line - 5),
                          (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, text_label, text_org, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)


face_detector = FaceDetect()
