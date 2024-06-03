
# PointColoring: 3D Object Detection with Sensor Fusion in the Driving Environment

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

---

## Features
- [x] Super fast and accurate 3D object detection using sensor fusion
- [x] Fast training, fast inference
- [x] Leverages LiDAR and RGB data
- [x] No Non-Max-Suppression needed
- [x] Support [distributed data parallel training](https://github.com/pytorch/examples/tree/master/distributed/ddp)
- [x] Release pre-trained models

## Highlights
- [x] The technical details are described **[here](./Technical_details.md)**
- [x] Introduction and explanation from _`Computer Vision and Perception for Self-Driving Cars Course`_ **[Youtube link](https://youtu.be/cPOtULagNnI?t=4858)**
- [x] PointColoring is used in the _`Udacity Self-Driving Car Engineer Nanodegree Program: Sensor Fusion and Tracking`_ **[GitHub link](https://github.com/udacity/nd013-c2-fusion-starter/tree/b1455b8ff433cb7f537d62e526209738293e7d8b)**

## Demonstration (on a single RTX A6000)

[![demo](http://img.youtube.com/vi/FI8mJIXkgX4/0.jpg)](http://www.youtube.com/watch?v=FI8mJIXkgX4)

**[Youtube link](https://youtu.be/FI8mJIXkgX4)**

## Getting Started

### Requirement

Instructions for setting up a virtual environment can be found [here](https://github.com/maudzung/virtual_environment_python3).

```shell
git clone https://github.com/ghshim/point-coloring.git PointColoring
cd PointColoring/
pip install -r requirements.txt
```

### Data Preparation

Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

The downloaded data includes:

- Velodyne point clouds _**(29 GB)**_
- Training labels of object dataset _**(5 MB)**_
- Camera calibration matrices of object dataset _**(16 MB)**_
- **Left color images** of object dataset _**(12 GB)**_ (for visualization purposes only)

Ensure that you structure the source code and dataset directories as follows:

### How to Run

#### Visualize the Dataset

To visualize 3D point clouds with 3D boxes, execute:

```shell
cd sfa/data_process/
python kitti_dataset.py
```

#### Inference

The pre-trained model is available in this repository.

```shell
python test.py --gpu_idx 0 --peak_thresh 0.2
```

#### Making Demonstration

```shell
python demo_2_sides.py --gpu_idx 0 --peak_thresh 0.2
```

The data for the demonstration will be automatically downloaded by executing the above command.

#### Training

##### Single Machine, Single GPU

```shell
python train.py --gpu_idx 0
```

#### Evaluation

To evaluate the model using a pre-trained checkpoint:

```shell
python train.py --evaluate --gpu_idx 0 --pretrained_path=../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth
```

#### Tensorboard

To track the training progress, go to the `logs/` folder and:

```shell
cd logs/<saved_fn>/tensorboard/
tensorboard --logdir=./
```

Then go to [http://localhost:6006/](http://localhost:6006/)

## Contact

If you find this work useful, please give it a star! <br>
If you find any errors or have suggestions, please contact us (**Email:** `asdfasdfm@unist.ac.kr`). <br>
Thank you!

## Citation

```bibtex
@misc{PointColoring,
  author =       {Seonghyeon Kim, Gahyeon Shim, Haechan Cheong},
  title =        {{PointColoring: 3D Object Detection with Sensor Fusion in the Driving Environment}},
  howpublished = {\url{https://github.com/ghshim/point-coloring}},
  year =         {2020}
}
```

## References

[1] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[2] RTM3D: [PyTorch Implementation](https://github.com/maudzung/RTM3D) <br>
[3] Libra_R-CNN: [PyTorch Implementation](https://github.com/OceanPang/Libra_R-CNN)

_The YOLO-based models with the same BEV maps input:_ <br>
[4] Complex-YOLO: [v4](https://github.com/maudzung/Complex-YOLOv4-Pytorch), [v3](https://github.com/ghimiredhikura/Complex-YOLOv3), [v2](https://github.com/AI-liu/Complex-YOLO)

*3D LiDAR Point pre-processing:* <br>
[5] VoxelNet: [PyTorch Implementation](https://github.com/skyhehe123/VoxelNet-pytorch)

## Folder Structure

```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/
        └── classes_names.txt
└── sfa/
    ├── config/
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   └── kitti_data_utils.py
    ├── models/
    │   ├── fpn_resnet.py
    │   ├── resnet.py
    │   └── model_utils.py
    └── utils/
    │   ├── demo_utils.py
    │   ├── evaluation_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── demo_2_sides.py
    ├── demo_front.py
    ├── test.py
    └── train.py
├── README.md 
└── requirements.txt
```

[python-image]: https://img.shields.io/badge/Python-3.6-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
```
