import cv2
import json
import os
import logging
import numpy as np


def get_data(input_path, cache=True, cache_path='./train_data_info.json'):
    if cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as fp:
                res = json.loads(fp.read())
            return res['all_data'], res['classes_count'], res['class_mapping']
        except Exception as e:
            logging.warning('Failed to load train_data_info, will generate this file again, '
                            'Exception: %s' % e)

    found_bg = False
    all_imgs = {}

    classes_count = {}  # 每个类别，对应的数量

    class_mapping = {}  # 每个类别名，对应ID

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = [i.strip() for i in line.split(',')]
            (filename, x1, y1, x2, y2, class_name) = line_split
            _, img_type = os.path.splitext(filename)
            if img_type not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            # check image file, ignore non-image samples
            img = cv2.imread(filename)
            if img is None:
                continue
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)),
                                                 'y1': int(float(y1)), 'y2': int(float(y2))})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        # 缓存以便下次训练
        res = dict(all_data=all_data, classes_count=classes_count, class_mapping=class_mapping)
        if cache:
            with open(cache_path, 'w') as fp:
                fp.write(json.dumps(res))
            logging.info('Succeed to generate the cache file of train_data_info')
        return all_data, classes_count, class_mapping
