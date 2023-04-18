# Pipe-BD implementation for neural architecture search

## baseline
The code implementation is based on ProxylessNAS (ICLR'19).

We modifiy "timm" from "Pytorch Image Models"( https://github.com/huggingface/pytorch-image-models ).

## How to run
- First you need to pretrain teachers
1. Cifar10: please refer to "cifar10_train_teacher.py"
2. ImageNet: plase refer to "timm/"

- For whole procedure: "scripts/"

- For measure speedups: "speedup/"

# Main implementation of Pipe-BD 
please refer to "arch_search.py" and "nas_manager.py"

# Custom timm modificiation
please refer to "timm/models/efficientnet.py"
`
