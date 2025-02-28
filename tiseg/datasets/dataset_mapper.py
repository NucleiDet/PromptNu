import copy
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from .ops import class_dict
import json
import torch

def read_image(path):
    _, suffix = osp.splitext(osp.basename(path))
    if suffix == '.tif':
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif suffix == '.npy':
        img = np.load(path)
    else:
        img = Image.open(path)
        img = np.array(img)

    return img

def gen_global_label(data_info):
    global_info_path = "./global_label/global_label_consep.json"
    with open(global_info_path, 'r') as file:
        global_infos = json.load(file)
    img_id = '_'.join(data_info['data_id'].split('_')[:2])

    color_mapping = {"deep purple": 0}
    size_mapping = {"small": 0, "medium": 1, "large": 2}
    density_mapping = {"densely packed": 0, "moderately dense": 1, "sparsely distributed": 2}
    arrange_mapping = {"columnar": 0, "scattered": 1, "irregular": 2, "parallel": 3, "peripheral": 4, "radial": 5}
    shape_mapping = {"elliptical/oval": 0, "irregular": 1, "elongated": 2, "spindle-shaped": 3, "spherical": 4}

    global_label = {}
    for info in global_infos:
        if info['id'] == [img_id]:
            global_label['color'] = [color_mapping[color] for color in info['color']]
            global_label['size'] = [size_mapping[size] for size in info['size']]
            global_label['density'] = [density_mapping[density] for density in info['density']]
            global_label['arrange'] = [arrange_mapping[arrange] for arrange in info['arrange']]
            global_label['shape'] = [shape_mapping[shape] for shape in info['shape']]

    return global_label

class DatasetMapper(object):

    def __init__(self, test_mode, *, processes):
        self.test_mode = test_mode

        self.processes = []
        for process in processes:
            class_name = process.pop('type')
            pipeline = class_dict[class_name](**process)
            self.processes.append(pipeline)

    def __call__(self, data_info):
        data_info = copy.deepcopy(data_info)

        img = read_image(data_info['file_name'])
        sem_gt = read_image(data_info['sem_file_name'])
        inst_gt = read_image(data_info['inst_file_name'])
        inst_color_gt = read_image(data_info['inst_color_file_name'])
        type_gt = read_image(data_info['type_file_name'])

        data_info['ori_hw'] = img.shape[:2]

        h, w = img.shape[:2]
        assert img.shape[:2] == sem_gt.shape[:2]

        global_label = gen_global_label(data_info)

        data = {
            'img': img,
            'sem_gt': sem_gt,
            'inst_gt': inst_gt,
            'inst_color_gt': inst_color_gt,
            'type_gt': type_gt,
            'seg_fields': ['sem_gt', 'inst_gt', "inst_color_gt", "type_gt"],
            'data_info': data_info,
            'global_label': global_label            
        }
        for process in self.processes:
            data = process(data)

        return data
