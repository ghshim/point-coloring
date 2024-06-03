# PointColoring: Real-Time 3D Object Detection with Sensor Fusion in the Driving  Environment

### Team Project in Principles of Deep Learning (2024 Spring)

We are highly motivated by https://github.com/maudzung/SFA3D

---

## Features

- [x]  **Real-time 3D Object and BEV (Birds-Eye-View) Detection**: Utilizes sensor fusion to enhance detection accuracy and reliability.
- [x]  **Fast Inference Speed**: Optimized for real-time processing.
- [x]  **6-Channel BEV Map**: Proposes a novel 6-channel BEV map for improved detection capabilities.
- [x]  **Pre-trained Models Available**: Offers pre-trained models to facilitate quick deployment and testing.


## Description

Abstract here


## Getting Started

### 1. Requirements

To set up the environment, use the provided conda environment file:

```bash
conda env create --f coloring.yml
```

### 2. Data Preparation

Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

The downloaded data includes:

- Velodyne point clouds ***(29 GB)***
- Training labels of object data set ***(5 MB)***
- Camera calibration matrices of object data set ***(16 MB)***
- **Left color images** of object data set ***(12 GB)*** (For visualization purposes only)

Please make sure that the source code and dataset directories are structured as follows:

```bash
${ROOT}
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
```

## How to Run

First, navigate to the `sfa` directory:

```bash
cd sfa
```

### 1. Training

To train the model, run:

```bash
python train.py --gpu_idx 0
```

### 2. Testing

The pre-trained model is released in this repo.

To test the model using a pre-trained checkpoint, run:

```bash
python test.py --gpu_idx 0 --pretrained_path ../checkpoints/fpn_resnet_18/best_point_coloring.pth
```

### 3. Evaluation

The pre-trained model is released in this repo.

To evaluate the model's performance, run:

```bash
python evaluate.py --pretrained_path ../checkpoints/fpn_resnet_18/best_point_coloring.pth
```

## Contact

If you find any errors or have any questions, feel free to contact us 🙂

[E-Mail]
