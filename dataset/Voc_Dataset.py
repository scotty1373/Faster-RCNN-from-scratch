# -*-  coding=utf-8 -*-
# @Time : 2022/3/24 17:22
# @Author : Scotty1373
# @File : Voc_Dataset.py
# @Software : PyCharm
import os
import time
import xml.etree.ElementTree as et
from utils_tools_dataset import get_image
import pandas as pd
import numpy as np


# Erase VocDataset difficult dataset
class VocBboxDataset:
    def __init__(self, data_root, *, idx_file='trainval'):
        """
        :param data_root:
        """
        self.root = os.path.join(data_root, 'VOC2007')
        idx_file_ = open(os.path.join(self.root, 'ImageSets',
                                      'Main', f'{idx_file}.txt')).readlines()
        self.idx_list = [i.rstrip('\n') for i in idx_file_]
        self.label_names = VOC_BBOX_LABEL_NAMES

    def get_item(self, idx):
        id_ = self.idx_list[idx]

        label = list()
        bbox = list()
        difficult = list()

        # Read xml tree structure
        annotation = et.parse(os.path.join(self.root, 'Annotations', f'{id_}.xml'))
        for anno in annotation.findall('object'):
            name = anno.find('name').text
            label.append(self.label_names.index(name))
            difficult.append(anno.find('difficult').text)
            bbox_info = anno.find('bndbox')
            xmin = bbox_info.find('xmin').text
            ymin = bbox_info.find('ymin').text
            xmax = bbox_info.find('xmax').text
            ymax = bbox_info.find('ymax').text
            bbox.append([xmin, ymin, xmax, ymax])

        img_path = os.path.join(self.root, 'JPEGImages', f'{id_}.jpg')
        img = get_image(img_path)

        # format to ndarray
        label = np.stack(label, axis=0).astype(np.int32)
        bbox = np.stack(bbox, axis=0).astype(np.float32)
        difficult = np.array(difficult, dtype=np.bool_).astype(np.uint8)

        return img, bbox, label, difficult


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


if __name__ == '__main__':
    dataset = VocBboxDataset('..\\Voc')
    dataset.get_item(1)
    time.time()