"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: The utils for evaluation
# Refer from: https://github.com/xingyizhou/CenterNet
"""

from __future__ import division
import os
import sys

import torch
import numpy as np
import torch.nn.functional as F
import cv2

src_dir = os.path.dirname(os.path.realpath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from tqdm import tqdm
from data_process.transformation import lidar_to_camera_box
import kitti_common
import box_np_ops


import config.kitti_config as cnf
from data_process.kitti_bev_utils import drawRotatedBox

def load_classes(path):


    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def post_processing_v2(outputs, conf_thresh=0.5, nms_thresh=0.5):
    """
    :param outputs: [batch_size, num_classes, num_anchors, 7]
    :param conf_thresh:
    :param nms_thresh:
    :return:
    """
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    detections = []
    for i in range(len(outputs)):
        output = outputs[i]
        for cls_id in range(output.shape[1]):
            cls_scores = _sigmoid(output[i, cls_id, :, 0])
            scores = cls_scores
            # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
            cls_detections = decode(output[i, cls_id, :, 0:1], output[i, cls_id, :, 1:3], output[i, cls_id, :, 3:5],
                                    output[i, cls_id, :, 5:6], output[i, cls_id, :, 6:9], K=40)
            # Filter out low scores
            keep_inds = (cls_detections[:, 0] > conf_thresh)
            cls_detections = cls_detections[keep_inds]
            scores = scores[keep_inds]
            # Apply NMS
            keep_inds = box_np_ops.boxes_iou_bev(cls_detections[:, 1:7], cls_detections[:, 1:7]) < nms_thresh
            cls_detections = cls_detections[keep_inds]
            scores = scores[keep_inds]
            # Add class index
            cls_detections = np.concatenate([cls_detections, np.full((len(cls_detections), 1), cls_id)], axis=1)
            # Add scores
            cls_detections = np.concatenate([cls_detections, scores[:, None]], axis=1)
            detections.append(cls_detections)
    return detections
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.floor_divide(topk_inds, width)).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.floor_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    batch_size, num_classes, height, width = hm_cen.size()

    # Apply non-maximum suppression (NMS) to the heatmap
    hm_cen = _nms(hm_cen)

    # Get top-K predictions from the heatmap
    scores, inds, clses, ys, xs = _topk(hm_cen, K=K)

    # Apply center offset to predicted x and y positions if available
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)
    print (detections.shape)

    return detections

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])


def post_processing(detections, num_classes=3, down_ratio=4, peak_thresh=0.2):
    """
    :param detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :return:
    """
    # TODO: Need to consider rescale to the original scale: x, y

    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / cnf.bound_size_y * cnf.BEV_WIDTH,
                detections[i, inds, 6:7] / cnf.bound_size_x * cnf.BEV_HEIGHT,
                get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]
        ret.append(top_preds)

    return ret


def draw_predictions(img, detections, num_classes=3):
    for j in range(num_classes):
        if len(detections[j]) > 0:
            for det in detections[j]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                drawRotatedBox(img, _x, _y, _w, _l, _yaw, cnf.colors[int(j)])
    return img

def convert_det_to_real_values(detections, num_classes=3, add_score=False):
    kitti_dets = []
    for cls_id in range(num_classes):
        if len(detections[cls_id]) > 0:
            for det in detections[cls_id]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                _yaw = -_yaw
                x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
                y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                z = _z + cnf.boundary['minZ']
                w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x
                if not add_score:
                    kitti_dets.append([cls_id, x, y, z, _h, w, l, _yaw])                    
                else:
                    kitti_dets.append([cls_id, x, y, z, _h, w, l, _yaw, _score])

    return np.array(kitti_dets)
def get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :6]
        pred_scores = output[:, 6]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                #iou, box_index = rotated_bbox_iou(pred_box.unsqueeze(0), target_boxes, 1.0, False).squeeze().max(0)
                ious = rotated_bbox_iou_polygon(pred_box, target_boxes)
                iou, box_index = torch.from_numpy(ious).max(0)

                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def convert_detection_to_kitti_annos(detection, dataset):
    print("Convert detection results to kitti format")
    class_names = ['Pedestrian', 'Car', 'Cyclist' ]
    annos = []
    assert len(dataset.sample_id_list) == len(detection)
    for i in tqdm(range(len(dataset.sample_id_list))):

        calib =  dataset.get_calib(dataset.sample_id_list[i])
        _, img = dataset.get_image(dataset.sample_id_list[i])
        
        kitti_lidar_dets = convert_det_to_real_values(detection[i],add_score=True)
        kitti_dets = np.array(kitti_lidar_dets)

        if len(kitti_dets) > 0:
            kitti_dets[:, 1:-1] = lidar_to_camera_box(kitti_lidar_dets[:, 1:-1], calib.V2C, calib.R0, calib.P2)
            locs = kitti_dets[:, 1:4]
            dims = kitti_dets[:, [6,4,5]]
            angles = kitti_dets[:, -2]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_np_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)
            box_corners_in_image = box_np_ops.project_to_image(
                box_corners, calib.P2)
            # box_corners_in_image: [N, 8, 2]
            minxy = np.min(box_corners_in_image, axis=1)
            maxxy = np.max(box_corners_in_image, axis=1)
            bbox = np.concatenate([minxy, maxxy], axis=1)


        anno = kitti_common.get_start_result_anno()
        num_example = 0
        for idx in range(kitti_dets.shape[0]):
            
            image_shape = img.shape[:2]
            if bbox[idx, 0] > image_shape[1] or bbox[idx, 1] > image_shape[0]:
                continue
            if bbox[idx, 2] < 0 or bbox[idx, 3] < 0:
                continue
            bbox[idx, 2:] = np.minimum(bbox[idx, 2:], image_shape[::-1])
            bbox[idx, :2] = np.maximum(bbox[idx, :2], [0, 0])            
  
            anno["bbox"].append(bbox[idx])
            anno["alpha"].append(-np.arctan2(-kitti_lidar_dets[idx, 3], kitti_lidar_dets[idx, 2])+kitti_dets[idx,-2])
            anno["dimensions"].append(kitti_dets[idx, [6,4,5]])
            anno["location"].append(kitti_dets[idx, 1:4])
            anno["rotation_y"].append(kitti_dets[idx,-2])

            anno["name"].append(class_names[int(kitti_dets[idx,0])])
            anno["truncated"].append(0.0)
            anno["occluded"].append(0)
            anno["score"].append(kitti_lidar_dets[idx,-1])
            num_example += 1
        if num_example != 0:
            anno = {n: np.stack(v) for n, v in anno.items()}
            annos.append(anno)
        else:
            annos.append(kitti_common.empty_result_anno())       
    return annos