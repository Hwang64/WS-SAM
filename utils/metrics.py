import torch
import numpy as np

def iou(mask1, mask2):
    intersection = (mask1 & mask2).float().sum((1, 2))
    union = (mask1 | mask2).float().sum((1, 2))
    return (intersection + 1e-6) / (union + 1e-6)

def precision_recall(preds, gts):
    preds = preds > 0.5
    gts = gts > 0.5
    TP = (preds & gts).float().sum()
    FP = (preds & ~gts).float().sum()
    FN = (~preds & gts).float().sum()
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return precision.item(), recall.item()

def compute_map(pred_boxes, gt_boxes):
    # Stub function; integrate pycocotools or VOC evaluator in real use
    return 0.75  # placeholder
