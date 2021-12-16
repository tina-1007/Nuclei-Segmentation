# Nuclei-Segmentation

This is my implement for [2021 VRDL HW2](https://codalab.lisn.upsaclay.fr/competitions/333?secret_key=3b31d945-289d-4da6-939d-39435b506ee5), which is an assignment to do instance segmentation on Nuclei Dataset.

I use toe toolbox [MMDetection](https://github.com/open-mmlab/mmdetection) to train my Mask R-CNN model.


## Installation

Install PyTorch and torchvision:
```
conda install pytorch torchvision -c pytorch
```
Install MMDetection:
```
pip install openmim
mim install mmdet
```
Or you can reference https://mmdetection.readthedocs.io/en/v2.19.1/get_started.html for more install tutorial.

## Prepare data
#### 1. Install relation package and go into `data` directory
```
pip install scikit-image
%cd data
```
#### 2. Choose some training data as validation data
```
data
|- test
|- train -> 22 image directories
  |- TCGA-18-5592-01Z-00-DX1
  ...
|- val   -> 2 image directories
  |- TCGA-NH-A8F7-01A-01-TS1
  |- TCGA-RD-A8N9-01A-01-TS1
```
#### 3. Rearrange the images to new directory
- Put all nuclei images to same directory
- For mask images, rename the file as `{image_filename}_{id}.png`
```
python rename_image.py
```
All images in `/images` and all masks in `/masks`

#### 4. Generate coco format annotation files
```
python rename_image.py
```
I also put my train and val annotation files in `/data/annotations`

---

The directory will looks like:
```
data
|- annotation  -> Location of annotation.json
|- new_train   -> Rearranged training data
|- new_val     -> Rearranged validation data
|- test
|- train
|- val
|- test_img_ids.json
|- to_coco.py
|- rename_image.py
```

## Testing
#### 1. Download the trained weights 
Get my trained model from [here](https://drive.google.com/file/d/18n7ma7Fxx_CtarzpzTDfWfNesJbptY0G/view?usp=sharing) and put it in root directory

#### 2. Inference
``` 
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/my_config.py  epoch_50.pth --format-only --options "jsonfile_prefix=./answer"
```
The `answer.segm.json` is the file to submit.

## Training

First, frozen a part of the model to train 15 epochs.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/my_config.py 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh {path_to_config_file} {GPUs_number}
```
Then, unfrozen the whole model to train another 35 epochs.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/my_config_2.py 4
```

## Reference

1. Mask R-CNN - [Paper](https://arxiv.org/abs/1703.06870) | [Github](https://github.com/matterport/Mask_RCNN)
2. MMDetection - [Github](https://github.com/open-mmlab/mmdetection)


