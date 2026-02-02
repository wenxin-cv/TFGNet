# TFGNet
This repository contains a pytorch implementation for the paper: TFGNet: Target Face Generation from Low-Quality Images via Textual Guidance.
## Dataset and Pre-trained Models
Please download Training Dataset ([FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)), then place it in the project root directory.

Please download pre-trained models ([Baidu Disk](https://pan.baidu.com/s/1QbvWxKWccj3PJgykGXjvRQ), code: gcyx), and then place……

## Environment
```bash
conda create -n tfg python=3.10  # (Python >= 3.8)
conda activate tfg
pip install -r requirements.txt
```
## Train
```bash
Stage 1: python train.py -opt options/stage1.yml
Stage 2: python train.py -opt options/stage2.yml
```
## Inference
```bash
python test.py
```
