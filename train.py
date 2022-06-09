from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

import os
import numpy as np
import pickle

from utils import *

def get_data_dicts(directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}

        filename = os.path.join(directory, img_anns["imagePath"])

        record["file_name"] = filename
        record["height"] = 1024
        record["width"] = 1024

        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]  # x coord
            py = [a[1] for a in anno['points']]  # y-coord
            poly = [(x, y) for x, y in zip(px, py)]  # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def train():
    config_file_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    output_dir = "./output/instance_seg"
    num_classes = 2
    classes = ['Cfos_positive', 'Cfos_neg']
    device = "cpu"

    cfg_save_path = "CI_cfg.pickle"

    # get the path to the data
    print("input data path")
    user_inp = input()
    data_path = user_inp
    if (user_inp == "def"):
        data_path = "/Users/tannerwatts/Desktop/serotonin-segmentation/"
    # register train and test dataset
    for d in ["train", "test"]:
        DatasetCatalog.register(
            "cells_" + d,
            lambda d=d: get_data_dicts(data_path + d, classes)
        )
        MetadataCatalog.get("cells_" + d).set(thing_classes=classes)
    microcontroller_metadata = MetadataCatalog.get("cells_train")

    # verify dataset
    plot_samples(dataset_name="cells_train", n=1)

