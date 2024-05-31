"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

from data_process.kitti_data_utils import Calibration
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.transformation import lidar_to_camera_box
import config.kitti_config as cnf
from utils.torch_utils import _sigmoid
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.misc import make_folder, time_synchronized
from models.model_utils import create_model
from data_process.kitti_bev_utils import drawRotatedBox
from data_process.kitti_dataset import KittiDataset
from data_process.kitti_dataloader import create_val_dataloader
import numpy as np
import torch
import cv2
from easydict import EasyDict as edict
import argparse
import sys
import os
import time
import warnings
import pickle
import matplotlib.image as mpimg

warnings.filterwarnings("ignore", category=UserWarning)


src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)


def parse_val_configs():
    parser = argparse.ArgumentParser(
        description='Testing config for the Implementation')
    parser.add_argument('--show', action='store_true', help='Show result or not')
    parser.add_argument('--save_bev', action='store_true', help='Draw prediction result on validation dataset')
    parser.add_argument('--draw_prediction', action='store_true', help='Draw prediction result on validation dataset')
    parser.add_argument('--draw_target', action='store_true', help='Draw target on validation dataset')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/best_point_coloring.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ############## Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join('../dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(
            configs.root_dir, 'results', 'validation')
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_val_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(
        configs.pretrained_path)
    model.load_state_dict(torch.load(
        configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device(
        'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()
    val_dataloader = create_val_dataloader(configs)
    if configs.save_test_output:
        make_folder(os.path.join(configs.results_dir, 'detect'))
        make_folder(os.path.join(configs.results_dir, 'detect', 'prediction'))
        make_folder(os.path.join(configs.results_dir, 'detect', 'target'))
        make_folder(os.path.join(configs.results_dir, 'bev'))

    if configs.draw_prediction:
        with torch.no_grad():
            fps = {}
            for batch_idx, batch_data in enumerate(val_dataloader):
                start_t = time.time()

                metadatas, bev_maps, targets = batch_data
        
                img_rgbs = [mpimg.imread(img_path) for img_path in metadatas['img_path']]
                batch_size = bev_maps.size(0)
                
                for k in targets.keys():
                    targets[k] = targets[k].to(configs.device, non_blocking=True)

                input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
                outputs = model(input_bev_maps)  # inferencing

                outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
                outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
                # detections size (batch_size, K, 10)
                detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                    outputs['dim'], K=configs.K)  # size ouput (batch_size, K, 10)
                detections = detections.cpu().numpy().astype(np.float32)
                detections = post_processing(
                    detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
                # t2 = time_synchronized()

                detections = detections[0]  # only first batch
                # Draw prediction in the image
                bev_map = (bev_maps.squeeze().permute(
                    1, 2, 0).numpy() * 255).astype(np.uint8)
                bev_map = bev_map[..., :3]
                bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
                # print(bev_mapp)
                bev_map = draw_predictions(
                    bev_map, detections.copy(), configs.num_classes)

                # Rotate the bev_map
                bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

                img_path = metadatas['img_path'][0]
                img_rgb = img_rgbs[0]
                img_rgb = img_rgb * 255
                img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                calib = Calibration(img_path.replace(
                    ".png", ".txt").replace("image_2", "calib"))
                kitti_dets = convert_det_to_real_values(detections)
                if len(kitti_dets) > 0:
                    kitti_dets[:, 1:] = lidar_to_camera_box(
                        kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                    img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

                out_img = merge_rgb_to_bev(
                    img_bgr, bev_map, output_width=configs.output_width)
                # print("kiti_dets: ", kitti_dets)
                
                end_t = time.time()
                
                inference_t = end_t - start_t
                
                print(f'Inference time: {inference_t:.5f}s')
                print(f'Done testing the {batch_idx}th sample, time: {inference_t*1000:.5f}ms, speed {1/inference_t:.2f}FPS')
                
                if configs.save_test_output:
                    if configs.output_format == 'image':
                        img_fn = os.path.basename(metadatas['img_path'][0])[:-4]
                        cv2.imwrite(os.path.join(configs.results_dir, 'detect', 'prediction',
                                    '{}.jpg'.format(img_fn)), out_img)
                        fps[img_fn] = inference_t
                    elif configs.output_format == 'video':
                        if out_cap is None:
                            out_cap_h, out_cap_w = out_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            out_cap = cv2.VideoWriter(
                                os.path.join(configs.results_dir, 'detect', 'prediction', '{}.avi'.format(
                                    configs.output_video_fn)),
                                fourcc, 30, (out_cap_w, out_cap_h))

                        out_cap.write(out_img)
                    else:
                        raise TypeError

                if configs.show:
                    cv2.imshow('test-img', out_img)
                    print(
                        '\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                    if cv2.waitKey(0) & 0xFF == 27:
                        break
        if out_cap:
            out_cap.release()
        if configs.show:
            cv2.destroyAllWindows()

    if configs.draw_target:
        # with torch.no_grad():
        #     for batch_idx, batch_data in enumerate(val_dataloader):
        #         metadatas, bev_maps, targets = batch_data
        #         print(bev_maps.shape)
        #         # batch_size = imgs.size(0)
        #         input_bev_maps = bev_maps.to(
        #             configs.device, non_blocking=True).float()
        #         for k in targets.keys():
        #             targets[k] = targets[k].to(configs.device, non_blocking=True)

        #         bev_map = (bev_maps.squeeze().permute(
        #             1, 2, 0).numpy() * 255).astype(np.uint8)
        #         bev_map = bev_map[..., :3]
        #         bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        #         targets_ = decode(targets['hm_cen'], targets['cen_offset'], targets['direction'], targets['z_coor'],
        #                         targets['dim'], K=configs.K)  # size ouput (batch_size, K, 10)
        #         targets_ = targets_.cpu().numpy().astype(np.float32)
        #         targets_ = post_processing(
        #             targets_, configs.num_classes, configs.down_ratio, configs.peak_thresh)
                
        #         bev_map = draw_predictions(
        #             bev_map, targets_.copy(), configs.num_classes)

        #         # Rotate the bev_map
        #         bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

        #         img_path = metadatas['img_path'][0]
        #         img_rgb = img_rgbs[0]
        #         img_rgb = img_rgb * 255
        #         img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
        #         img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        #         calib = Calibration(img_path.replace(
        #             ".png", ".txt").replace("image_2", "calib"))
        #         kitti_dets = convert_det_to_real_values(detections)
        #         if len(kitti_dets) > 0:
        #             kitti_dets[:, 1:] = lidar_to_camera_box(
        #                 kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
        #             img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

        #         out_img = merge_rgb_to_bev(
        #             img_bgr, bev_map, output_width=configs.output_width)
        #         # print("kiti_dets: ", kitti_dets)
                
        #         end_t = time.time()
                
        #         inference_t = end_t - start_t
                
        #         print(f'Inference time: {inference_t:.5f}s')
        #         print(f'Done testing the {batch_idx}th sample, time: {inference_t*1000:.5f}ms, speed {1/inference_t:.2f}FPS')
                
        #         if configs.show:
        #             cv2.imshow('bev_map', out_img)
        #             if cv2.waitKey(0) & 0xff == 27:
        #                 break

        #         if configs.save_test_output:
        #             if configs.output_format == 'image':
        #                 img_fn = os.path.basename(img_path)[:-4]
        #                 cv2.imwrite(os.path.join(configs.results_dir, 'detect', 'target',
        #                             '{}.jpg'.format(img_fn)), out_img)
                        
        #             elif configs.output_format == 'video':
        #                 if out_cap is None:
        #                     out_cap_h, out_cap_w = out_img.shape[:2]
        #                     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #                     out_cap = cv2.VideoWriter(
        #                         os.path.join(configs.results_dir, 'detect', 'target', '{}.avi'.format(
        #                             configs.output_video_fn)),
        #                         fourcc, 30, (out_cap_w, out_cap_h))

        #                 out_cap.write(out_img)
        #             else:
        #                 raise TypeError
        # if out_cap:
        #     out_cap.release()
        # if configs.show:
        #     cv2.destroyAllWindows()
        
        # def get_bounding_box_corners(target):
        #     # target 데이터에서 필요한 정보 추출
        #     cat_id, x, y, z, h, w, l, ry = target

        #     # bounding box의 꼭지점 계산
        #     corners = np.array([
        #         [-l/2, -w/2],  # front-left
        #         [l/2, -w/2],   # front-right
        #         [l/2, w/2],    # back-right
        #         [-l/2, w/2],   # back-left
        #         [-l/2, -w/2]   # front-left (close the box)
        #     ])

        #     # bounding box 회전
        #     rotation_matrix = np.array([
        #         [np.cos(ry), -np.sin(ry)],
        #         [np.sin(ry), np.cos(ry)]
        #     ])
        #     rotated_corners = np.dot(corners, rotation_matrix.T)

        #     # bounding box 이동
        #     rotated_corners[:, 0] += x
        #     rotated_corners[:, 1] += y

        #     return rotated_corners

        # def get_bounding_box_points(target):
        #     corners = get_bounding_box_corners(target)

        #     # bounding box의 두 점 계산
        #     min_x = np.min(corners[:, 0])
        #     max_x = np.max(corners[:, 0])
        #     min_y = np.min(corners[:, 1])
        #     max_y = np.max(corners[:, 1])

        #     return (min_x, min_y), (max_x, max_y)
        
        # def draw_bounding_boxes_on_bev(bev_image, targets):
        #     colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
        #               [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]
            
        #     for target in targets:
        #         cat_id = int(target[0])
        #         (min_x, min_y), (max_x, max_y) = get_bounding_box_points(target)
        #         bev_image = cv2.rectangle(bev_image, (min_x, min_y), (max_x, max_y), colors[cat_id], 2)
            
        #     return bev_image
    
        val_dataset = KittiDataset(configs, mode='val', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples)
    
        for idx in range(len(val_dataset)):
            # Here you can obtain predictions using your model
            bev_map, labels, img_rgb, img_path = val_dataset.draw_img_with_label(idx)
            
            if configs.save_bev:
                img_fn = os.path.basename(img_path)[:-4]
                npy_path = os.path.join(configs.results_dir, 'bev', '{}.npy'.format(img_fn))
                np.save(npy_path, bev_map)

            calib = Calibration(img_path.replace(
                ".png", ".txt").replace("image_2", "calib"))
            bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
            bev_map = bev_map[..., :3]
            bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
            
            for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
                # Draw rotated box
                yaw = -yaw
                y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
                x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
                w1 = int(w / cnf.DISCRETIZATION)
                l1 = int(l / cnf.DISCRETIZATION)

                drawRotatedBox(bev_map, x1, y1, w1, l1,
                               yaw, cnf.colors[int(cls_id)])
           
            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            
            labels[:, 1:] = lidar_to_camera_box(
                labels[:, 1:], calib.V2C, calib.R0, calib.P2)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_bgr = show_rgb_image_with_boxes(img_bgr, labels, calib)
            
            out_img = merge_rgb_to_bev(
                img_bgr, bev_map, output_width=configs.output_width)
            
            if configs.show:
                cv2.imshow('bev_map', out_img)
                if cv2.waitKey(0) & 0xff == 27:
                    break

            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(img_path)[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, 'detect', 'target',
                                '{}.jpg'.format(img_fn)), out_img)
                    
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, 'detect', 'target', '{}.avi'.format(
                                configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError
