import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image
from matting.sbbm_segmentation import perform_SBBM_segmentation
from matting.ocsvm_segmentation import perform_ocsvm_segmentation
from matting.alpha_matting_segmentation import perform_alpha_matting

MATTING_DICT = {'sbbm': perform_SBBM_segmentation,
               'ocsvm': perform_ocsvm_segmentation,
               'lbdm': perform_alpha_matting}


def matting_select(image_path, target_bbox, matting_method, logdir):
    """
    Args:
        image_path: 绝对路径
        target_bbox: 初始帧box坐标：[y0, x0, h, w]

    Returns: 第一帧抠图图像路径
    """
    image = Image.open(image_path)
    image = np.array(image)
    bbox = np.array([target_bbox[1], target_bbox[0],
                     target_bbox[1] + target_bbox[3], target_bbox[0],
                     target_bbox[1] + target_bbox[3], target_bbox[0] + target_bbox[2],
                     target_bbox[1], target_bbox[0] + target_bbox[2]])

    mask = MATTING_DICT[matting_method](image, bbox)
    # remove pixels from image that are labelled as background
    image_masked = image.copy()
    for d in range(3):
        image_masked[..., d].flat[~mask.ravel()] = 0

    # save matting to file
    matting = Image.fromarray(image_masked)

    matting_dir = logdir + '/matted'
    if not os.path.exists(matting_dir):
        os.makedirs(matting_dir)
    matting_path = os.path.join(matting_dir, os.path.basename(image_path)[:-4] + '_matted.jpg')
    logging.info('The path of matting image saving is %s'%matting_path)
    matting.save(matting_path)
    return matting_path

if __name__ == '__main__':

    CURRENT_PATH = os.getcwd()

    image_path = CURRENT_PATH + '/images/crossing_00000066.jpg'
    bbox = [261.45, 495.64,
            445.24 - 261.45, 532.31 - 495.64]

    matting_path = matting_select(image_path, bbox, matting_method='sbbm', logdir=os.path.dirname(image_path))
    print(matting_path)