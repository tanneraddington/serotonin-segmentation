import torch
assert torch.__version__.startswith("1.8")
import torchvision
import cv2

import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer


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

def train(data_path, should_train):
    '''
    This method trains the model on the labeled images that were specified in get_data_dicts

    :return:
    '''
    # the goal here is to resuse this method without having to re train each iteration
    cfg = get_cfg()
    file_exists = os.path.exists(data_path+'/model_final.pth')
    if file_exists and should_train == "F":
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TEST = ("test",)
        predictor = DefaultPredictor(cfg)
        return predictor

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("category_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # change this when you can use a gpu
    # cfg.MODEL.DEVICE = "cpu"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("test",)
    predictor = DefaultPredictor(cfg)
    return predictor


def main():
    '''
    The main method generates the data, and then
    :return:
    '''
    #setup the data
    classes = ['Cfos_positive', 'Cfos_neg']
    print("input data path")
    data_path = input()
    print("Need To Train? (T) for true (F) for false")
    should_train = input()
    for d in ["train", "test"]:
        DatasetCatalog.register(
            "category_" + d,
            lambda d=d: get_data_dicts(data_path + d, classes)
        )
        MetadataCatalog.get("category_" + d).set(thing_classes=classes)
    microcontroller_metadata = MetadataCatalog.get("category_train")
    predictor = train()
    print("INPUT IMAGE PATH")
    test_dataset_dicts = get_data_dicts(data_path + 'test', classes)

    # ask user for image path
    image_path = input()
    image = cv2.imread(d[image_path])
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1],
                   metadata=microcontroller_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW)  # removes the colors of unsegmented pixels

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()