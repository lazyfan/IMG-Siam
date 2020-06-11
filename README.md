# IMG-Siam
A TensorFlow implementation of the IMG-Siam tracker

## Introduction

This is a TensorFlow implementation of [Initial Matting-Guided Visual Tracking with Siamese Network](  https://ieeexplore.ieee.xilesou.top/stamp/stamp.jsp?tp=&arnumber=8674549  ). The code is improved on the TensorFlow version of SiamFC [here]( https://github.com/bilylee/SiamFC-TensorFlow ).

## Prerequisite
### Configuration environment

You can use TensorFlow > 1.0 for tracking though. Note the tracking performance slightly varies in different versions.

```bash
# pip install tensorflow    # For CPU
pip install tensorflow-gpu  # For GPU

# Other main modules
pip install sacred, scipy, opencv-python

# Matting needs
pip install scikit-image, scikit-learn, vlfeat-ctypes

# (OPTIONAL) Install nvidia-ml-py for automatically selecting GPU
pip install nvidia-ml-py
```

### Clone this repository to your disk

```bash
git clone https://github.com/lazyfan/IMG-Siam.git
```



## Tracking

In the initialization phase of the tracker, matting is performed on the initial frame.

### Run the tracker on the specified sequence

You can place the sequence you want to track in the `assets`, where the sequence *KiteSurf* is placed for reference.

(OPTIONAL) There are three matting programs available: sbbm, lbdm, ocsvm, you can modify it in `configuration.py`

```bash
python run_IMGSiam_tracker.py
```

You can modify `visualization` in the `configuration.py`, visualize the tracking results, or run it directly:

```
python scripts/show_tracking.py
```



## Training

On the basis of SiamFC, attention module is added to the model, named SiamAtt in paper. The training steps are as follows:

### 1. Download dataset

Download and unzip the ImageNet VID 2015 dataset (~86GB) [here](http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz).

### 2. Preprocess training data

```bash
python scripts/preprocess_VID_data.py
# Split train/val dataset and store corresponding image paths
python scripts/build_VID2015_imdb.py
```

### 3. Start training

(OPTIONAL) There are two attention modules available: se_block, cbam_block, you can modify it in `configuration.py`, se_block by default.

#### 3.1 Start from scratch

```bash
python train_SiamAtt.py
```

#### 3.2 Load SiamFC pretrained variables & Fine tune the later two layers

Download pretrained models.

```bash
python scripts/download_assets.py
```

Convert pretrained MatConvNet model into TensorFlow format.

```bash
# Note we use SiamFC-3s-color-pretrained as one example. You
# Can also use SiamFC-3s-gray-pretrained. 
python convert_pretrained_model.py
```

Modify trainable variable scope in `train_SiamAtt.py` and start train.

```bash
python train_SiamAtt.py
```

### 4. View the training progress in TensorBoard

```bash
# Open a new terminal session and cd to IMG-Siam, then
tensorboard --logdir=Logs/track_model_checkpoints/IMGSiam-3s-color
```



## Reference

#### Paper

[1] [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549) 

[2] [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507) 

#### Repositories

[1] [SiamFC tensorflow implementation](https://github.com/bilylee/SiamFC-TensorFlow)

[2] [initialisation-problem](https://github.com/georgedeath/initialisation-problem)

[3] [CBAM-TensorFlow-Slim](https://github.com/kobiso/CBAM-tensorflow-slim)

