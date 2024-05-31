import tqdm
import os
import re
import numpy as np
import random
import json
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm
from collections import Counter

import torch
from utils.torch_utils import _sigmoid
from torch.utils.data import DataLoader, Dataset

from utils import kitti_common
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values, convert_detection_to_kitti_annos
from losses.losses import Compute_Loss
from utils.misc import AverageMeter
from utils.torch_utils import reduce_tensor, to_python_float
from models.model_utils import create_model
from data_process.kitti_dataloader import create_val_dataloader
from utils.eval import get_official_eval_result

def parse_eval_configs():
    parser = argparse.ArgumentParser(
        description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--classnames-infor-path', type=str, default='../../dataset/kitti/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
    # parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
    #                     help='The name of the model architecture')
    parser.add_argument('--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default='/home/ahrilab/PrincipleDL/point-coloring/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for IoU')

    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ############## Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')
    configs.head_conv = 256 if 'dla' in configs.arch else 64
    configs.imagenet_pretrained = False
    configs.num_classes = 3
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
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    
    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4
    configs.dataset_dir ='../dataset/kitti'
    return configs


def validate(val_dataloader, model, configs):
    losses = AverageMeter('Loss', ':.4e')
    criterion = Compute_Loss(device=configs.device)

    model.eval()
    detections_list = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader)):
            metadatas, imgs, targets = batch_data
            batch_size = imgs.size(0)
            for k in targets.keys():
                targets[k] = targets[k].to(configs.device, non_blocking=True)
            imgs = imgs.to(configs.device, non_blocking=True).float()
            outputs = model(imgs)
            total_loss, loss_stats = criterion(outputs, targets)
            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.gpu_idx is None):
                total_loss = torch.mean(total_loss)

            if configs.distributed:
                reduced_loss = reduce_tensor(
                    total_loss.data, configs.world_size)
            else:
                reduced_loss = total_loss.data
            losses.update(to_python_float(reduced_loss), batch_size)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])

            for idx in range(outputs['hm_cen'].shape[0]):
                # detections size (batch_size, K, 10)
                detections = decode(outputs['hm_cen'][idx:idx+1, :], outputs['cen_offset'][idx:idx+1, :],
                                    outputs['direction'][idx:idx+1,
                                                         :], outputs['z_coor'][idx:idx+1, :],
                                    outputs['dim'][idx:idx+1, :], K=configs.K)
                detections = detections.cpu().numpy().astype(np.float32)
                detections = post_processing(
                    detections, configs.num_classes, configs.down_ratio)
                detections_list.append(detections[0])

        dt_annos = convert_detection_to_kitti_annos(
            detections_list, val_dataloader.dataset)
        gt_annos = kitti_common.get_label_annos(
            val_dataloader.dataset.label_dir, val_dataloader.dataset.sample_id_list)
        print("Doing evaluation")
        result = get_official_eval_result(gt_annos, dt_annos,
              [i for i in range(configs.num_classes)])
        print(result)

    return losses.avg, result


def print_best_models(data, title):
    print(f"{title}")
    print(f"{'':<12} Hard         Moderate     Easy")
    for category in ['Car', 'Pedestrian', 'Cyclist']:
        hard = data['hard'][category]
        moderate = data['moderate'][category]
        easy = data['easy'][category]
        print(f"{category:<12} {hard:<12} {moderate:<12} {easy:<12}")

def print_results(data):
    print("[Result]\n")
    print(f"{'':<32}{'Hard':<20}{'Moderate':<20}{'Easy':<20}")
    for category, values in data.items():
        print(f"{category}")
        print("2D Bounding Box:", np.round(values['mAPbbox'], 2))
        print("Bird's-Eye View:", np.round(values['mAPbev'], 2))
        print("3D Bounding Box:", np.round(values['mAP3d']), 2)
        print("\n")

def find_best_model(classes, eval_result):
    best_mAP = {'hard':{}, 'moderate':{}, 'easy':{}}
    best_2d_epoch = {'hard':{}, 'moderate':{}, 'easy':{}}
    best_bev_epoch = {'hard':{}, 'moderate':{}, 'easy':{}}
    best_3d_epoch = {'hard':{}, 'moderate':{}, 'easy':{}}

    for c in classes:
        best_mAP['hard'][c] = {'2d':0, 'bev':0,'3d':0}
        best_mAP['moderate'][c] = {'2d':0, 'bev':0,'3d':0}
        best_mAP['easy'][c] = {'2d':0, 'bev':0,'3d':0}
    
    for epoch in eval_result.keys():
        result = eval_result[epoch]
        
        for c in classes:
            '''2D bounding box'''
            # Hard, 2D bounding box
            if result[c]['mAPbbox'][0] > best_mAP['hard'][c]['2d']:
                best_mAP['hard'][c]['2d'] = result[c]['mAPbbox'][0]
                best_2d_epoch['hard'][c] = epoch
            # Moderate, 2D bounding box
            if result[c]['mAPbbox'][1] > best_mAP['moderate'][c]['2d']:
                best_mAP['moderate'][c]['2d'] = result[c]['mAPbbox'][1]
                best_2d_epoch['moderate'][c] = epoch
            # Easy, 2D bounding box
            if result[c]['mAPbbox'][2] > best_mAP['easy'][c]['2d']:
                best_mAP['easy'][c]['2d'] = result[c]['mAPbbox'][2]
                best_2d_epoch['easy'][c] = epoch
            '''BEV'''
            # Hard, BEV
            if result[c]['mAPbev'][0] > best_mAP['hard'][c]['bev']:
                best_mAP['hard'][c]['bev'] = result[c]['mAPbev'][0]
                best_bev_epoch['hard'][c] = epoch
            # Moderate, BEV
            if result[c]['mAPbev'][1] > best_mAP['moderate'][c]['bev']:
                best_mAP['moderate'][c]['bev'] = result[c]['mAPbev'][1]
                best_bev_epoch['moderate'][c] = epoch
            # Easy, BEV
            if result[c]['mAPbev'][2] > best_mAP['easy'][c]['bev']:
                best_mAP['easy'][c]['bev'] = result[c]['mAPbev'][2]
                best_bev_epoch['easy'][c] = epoch
            '''3D bounding box'''
            # Hard, 3D bounding box
            if result[c]['mAP3d'][0] > best_mAP['hard'][c]['3d']:
                best_mAP['hard'][c]['3d'] = result[c]['mAP3d'][0]
                best_3d_epoch['hard'][c] = epoch
            # Moderate, 3D bounding box
            if result[c]['mAP3d'][1] > best_mAP['moderate'][c]['3d']:
                best_mAP['moderate'][c]['3d'] = result[c]['mAP3d'][1]
                best_3d_epoch['moderate'][c] = epoch
            # Easy, 3D bounding box
            if result[c]['mAP3d'][2] > best_mAP['easy'][c]['3d']:
                best_mAP['easy'][c]['3d'] = result[c]['mAP3d'][2]
                best_3d_epoch['easy'][c] = epoch

    print("best mAP\n", best_mAP)
    # print("best 2d bounding box model\n",best_2d_epoch)
    # print("best bev model\n",best_bev_epoch)
    # print("best 3d bounding box model\n",best_3d_epoch)
    
    print_best_models(best_2d_epoch, "Best 2D Bounding Box Model")
    print()
    print_best_models(best_bev_epoch, "Best BEV Model")
    print()
    print_best_models(best_3d_epoch, "Best 3D Bounding Box Model")
   
    # Combine all epochs into a single list
    all_epochs = []

    # Function to collect epochs from the data
    def collect_epochs(data):
        for difficulty in data.values():
            for epoch in difficulty.values():
                all_epochs.append(epoch)

    # Collect epochs from all the models
    collect_epochs(best_2d_epoch)
    collect_epochs(best_bev_epoch)
    collect_epochs(best_3d_epoch)

    # Calculate the frequency of each epoch
    epoch_counts = Counter(all_epochs)

    print("The number of epoch", epoch_counts)
    top_3_epochs = [epoch for epoch, count in sorted(epoch_counts.items(), key=lambda item: item[1], reverse=True)[:3]]

    # print("Top 3 epochs with the lowest frequency:")
    # for epoch in top_3_epochs:
    #     print(epoch)
    #     print_results(eval_result[epoch])

    # print()

    
    best = top_3_epochs[0]
    print(f"Best Model: epoch {best}")
    print("Brid's-Eye View")
    # print(f"{'':<32}{'Hard':<20}{'Moderate':<20}{'Easy':<20}")
    for category, values in eval_result[best].items():
        print(f"{category}", end='')
        # print("2D Bounding Box:", np.round(values['mAPbbox'], 2))
        print( np.round(values['mAPbev'], 2))
        # print("3D Bounding Box:", np.round(values['mAP3d']), 2)
    print("\n")
    print("3D Bounding Box")
    # print(f"{'':<32}{'Hard':<20}{'Moderate':<20}{'Easy':<20}")
    for category, values in eval_result[best].items():
        print(f"{category}", end='')
        # print("2D Bounding Box:", np.round(values['mAPbbox'], 2))
        # print("Bird's-Eye View:", np.round(values['mAPbev'], 2))
        print(np.round(values['mAP3d'], 2))
        # print("\n")



# 정규표현식을 사용하여 숫자를 추출하는 함수
def extract_epoch_number(file_path):
    return int(re.search(r'\d+', file_path).group())

def main():    
    configs = parse_eval_configs()
    # # Re-produce results
    # if configs.gpu_idx is not None:
    #     print(
    #         'You have chosen a specific GPU. This will completely disable data parallelism.')
    
    # configs.device = torch.device(
    #     'cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))

    # if not configs.distributed:
    #     configs.subdivisions = int(64 / configs.batch_size)

    # configs.is_master_node = (not configs.distributed) or (
    #     configs.distributed and (configs.rank % configs.ngpus_per_node == 0))
    classes = ['Car', 'Pedestrian', 'Cyclist']

    with open("../results/eval_result_1.json", "r") as json_file:
        eval_result = json.load(json_file)

    find_best_model(classes, eval_result)
    return

    pretrained_dir = '../checkpoints/fpn_resnet_18'
    pretrained_path = os.listdir(pretrained_dir)

    pattern = re.compile(r'\d+')
    pretrained_models = []
    for path in pretrained_path:
        if path.split('_')[0] == 'Model' and int(pattern.findall(path)[-1]) >= 298:
            pretrained_models.append(os.path.join(pretrained_dir, path))

    pretrained_models = sorted(pretrained_models, key=lambda x: int(pattern.findall(x)[-1]))

    val_dataloader = create_val_dataloader(configs)
    print('number of batches in val_dataloader: {}'.format(
                len(val_dataloader)))
    model = create_model(configs)

    eval_dict = {}
    for pretrained_model_path in pretrained_models:
        # pretrained_model_path = os.path.join(pretrained_dir, 'fpn_resnet_18_epoch_300.pth')
        model_name = os.path.basename(pretrained_model_path)
        print(type(torch.load(pretrained_model_path, map_location='cpu')))
        model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        model.to(configs.device)
        print('Loaded weights from {}\n'.format(pretrained_model_path))
        val_loss, result = validate(val_dataloader, model, configs)
        eval_dict[model_name] = result
        print('val_loss: {:.4e}'.format(val_loss))
        

    print(eval_dict)
    with open("../result.json", "w") as json_file:
        json.dump(eval_dict, json_file, indent=4)

if __name__ == '__main__':
    main()