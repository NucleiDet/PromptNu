# Enviroment Installation

This guide outlines the exact versions of Python, CUDA, and PyTorch, along with download links and installation commands.

## Setting up for CUDA 11.3

1. **Download and install CUDA 11.3:**  
   [CUDA 11.3 Download](https://developer.nvidia.com/cuda-11.3.0-download-archive)

2. **Download and install Anaconda:**  
   [Anaconda for Linux](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.1.0-Linux-x86_64.sh)

3. **Create and activate a new Conda environment:**
   ```bash
   conda create -n promptnu python=3.8 -y
   conda activate promptnu

4. **Install PyTorch:**
   ```bash
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

5. **Install MMCV:**
   ```bash
   pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

6. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt

7. **Install the tiseg:**
   ```bash
   pip install -e .

8. **Install the CLIP module:**
   ```bash
   cd CLIP && pip install -e .

## Setting up for CUDA 11.1

The project is also compatible with CUDA 11.1. The corresponding steps for setting up the environment are listed below:
1. **Download and install CUDA 11.1:**  
   [CUDA 11.1 Download](https://developer.nvidia.com/cuda-11.1.0-download-archive)

2. **Download and install Anaconda:**  
   [Anaconda for Linux](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.1.0-Linux-x86_64.sh)

3. **Create and activate a new Conda environment:**
   ```bash
   conda create -n promptnu python=3.8 -y
   conda activate promptnu

4. **Install PyTorch:**
   ```bash
   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch\_stable.html

5. **Install MMCV:**
   ```bash
   pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

6. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt

7. **Install the tiseg:**
   ```bash
   pip install -e .

8. **Install the CLIP module:**
   ```bash
   cd CLIP && pip install -e .


