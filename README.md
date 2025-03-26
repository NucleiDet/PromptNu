# Prompting Vision-Language Model for Nuclei Instance Segmentation and Classification

## Requirements
- CUDA 11.3
- Python 3.8.x
- PyTorch 1.11.0

## Installation
1. Install MMCV-full (Linux recommend): `pip install MMCV-full==1.3.13`;
2. Install requirements package: `pip install -r requirements.txt`;
3. Install tiseg: `pip install -e .`;

## Dataset Prepare 
Please check this doc

## Usage
### Training 
```bash
# single gpu training
python tools/train.py [config_path]
# multiple gpu training
./tools/dist_train.sh [config_path] [num_gpu]
# demo (cdnet for CPM17 dataset on 1 gpu)
python tools/train.py configs/unet/unet_vgg16_radam-lr5e-4_bs16_256x256_7k_cpm17.py
# demo (unet for CPM17 dataset on 4 gpu)
./tools/dist_train.py configs/unet/unet_vgg16_radam-lr5e-4_bs16_256x256_7k_cpm17.py 4
```

### Evaluation
```bash
# single gpu evaluation
python tools/test.py [config_path]
# multiple gpu evaluation
./tools/dist_test.py [config_path] [num_gpu]
```

