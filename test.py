from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *


def main():
    # use different pickle files for different models
    cfg_save_path = "CI_cfg.pickle"
    with open(cfg_save_path, 'rb') as f:
        cfg = pickle.load(f)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)

    image_path = "/Users/tannerwatts/Desktop/serotonin-segmentation/test/m5_1_6_2(4).jpg"
    predict_image(image_path, predictor)



if __name__ == '__main__':
    main()


