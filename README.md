# **PromptNu**

The official PyTorch implementation of the TMI paper:  
**[Prompting Vision-Language Model for Nuclei Instance Segmentation and Classification](https://ieeexplore.ieee.org/document/11050438)**

**Authors**: Jieru Yao, Guangyu Guo, Zhaohui Zheng, Qiang Xie, Longfei Han, Dingwen Zhang, Junwei Han.

![Model Framework](https://github.com/NucleiDet/PromptNu/blob/master/img/framework.jpg?raw=true)  
*Figure 1: Overview of the proposed model architecture.*

## **Abstract**
In this work, we introduce **PromptNu**, a novel framework designed to incorporate abundant nuclei knowledge into the training of nuclei instance recognition models. By leveraging **vision-language contrastive learning** and **prompt engineering techniques**, our method significantly improves the performance of nuclei instance segmentation and classification tasks. Extensive experiments conducted across six datasets, including various Whole Slide Imaging (WSI) scenarios, demonstrate the effectiveness of our approach.

![Results Visualization](https://github.com/NucleiDet/PromptNu/blob/master/img/visualization.jpg?raw=true)  
*Figure 2: Visualization of results on various datasets.*

## **Requirements**
Ensure your environment is set up with the following dependencies:

- **CUDA**: 11.3
- **Python**: 3.8.x
- **PyTorch**: 1.11.0

## Installation
Follow these steps to set up the environment and install necessary dependencies:

1. Install MMCV-full (Linux recommend): `pip install MMCV-full==1.3.13`;
2. Install requirements package: `pip install -r requirements.txt`;
3. Install tiseg: `pip install -e .`;
4. Install CLIP: `cd CLIP & pip install -e .`.

For the specific versions of Python, CUDA, and PyTorch, along with download links and installation commands, please refer to the [Environment Installation](https://github.com/NucleiDet/PromptNu/blob/master/docs/environment_installation.md) folder.

Alternatively, you can use our pre-built Docker image, which includes all necessary code and dependencies. Download and run the container directly from [Docker](https://drive.google.com/file/d/1S_T9LeEPnqoWpjlWhypS8wgv-xFyGYm6/view?usp=sharing)

## Dataset Prepare 
Please check this 
Please refer to the [Dataset Preparation](./docs/data_prepare.md) for detailed instructions on how to prepare the datasets.

## Usage
### Training 
```bash
# single gpu training
python tools/train.py [config_path]
# multiple gpu training
./tools/dist_train.sh [config_path] [num_gpu]
# demo (promptnu for MoNuSeg dataset on 1 gpu)
python tools/train.py configs/promptnu/promptnu_adam-lr1e-4_bs8_256x256_300e_monuseg.py
# demo (promptnu for MoNuSeg dataset on 4 gpu)
./tools/dist_train.py configs/promptnu/promptnu_adam-lr1e-4_bs8_256x256_300e_monuseg.py 4
```

### Evaluation
```bash
# single gpu evaluation
python tools/test.py [config_path]
# multiple gpu evaluation
./tools/dist_test.py [config_path] [num_gpu]
```

## Citation
If you find this repository useful for your research, please cite our work as follows:

```
@article{yao2025promptnu,
  author={Yao, Jieru and Guo, Guangyu and Zheng, Zhaohui and Xie, Qiang and Han, Longfei and Zhang, Dingwen and Han, Junwei},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Prompting Vision-Language Model for Nuclei Instance Segmentation and Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2025.3579214}}

```
