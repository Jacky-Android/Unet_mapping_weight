from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from mpl_toolkits.axes_grid1 import make_axes_locatable
 
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]) #分别对应通道 R G B
def weight_add(path):
    gt = io.imread(path)

    gt = 1 * (gt > 0)
    
    # 【1】计算细胞和背景的像素频率
    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())
    
    # 【2】归一化
    c_weights /= c_weights.max()
    
    # 【3】得到c_w字典
    c_weights.tolist()
    cw = {}
    for i in range(len(c_weights)):
        cw[i]=c_weights[i]
    weightMap_ = UnetWeightMap(gt, cw)
    return weightMap_

def UnetWeightMap(mask, wc=None, w0=10, sigma=5):
 
    mask_with_labels = label(mask)
    no_label_parts = mask_with_labels == 0
    label_ids = np.unique(mask_with_labels)[1:]
 
    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))
        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(mask_with_labels != label_id)
        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        weight_map = w0 * np.exp(-1/2 * ((d1+d2)/sigma) ** 2) * no_label_parts
        weight_map = weight_map + np.ones_like(weight_map)
 
        if wc is not None:
            class_weights = np.zeros_like(mask)
            for k, v in wc.items():
                class_weights[mask == k] = v
            weight_map = weight_map + class_weights
 
    else:
        weight_map = np.zeros_like(mask)
    return weight_map
 

 
