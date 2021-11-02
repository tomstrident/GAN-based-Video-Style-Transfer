# GAN-based-Video-Style-Transfer
Source files of my master thesis at TU Graz [GAN-based-Video-Style-Transfer](https://drive.google.com/file/d/1O2scchvoUWZtw6c60DvVeOOd4fiZLOBQ/view?usp=sharing).

## Methods:
In this work we performed a comparison between various style transfer methods and deviations.

| Year | Paper | Temporal | Multi-Domain | Code |
| :----: | ----- | :----: | :----: | :----: |
| 2015 | [Image style transfer using convolutional neural networks](https://arxiv.org/abs/1508.06576) | :heavy_multiplication_x: | :heavy_multiplication_x: | [Code](https://github.com/leongatys/PytorchNeuralStyleTransfer) |
| 2016 | [Artistic Style Transfer for Videos](https://arxiv.org/abs/1604.08610) | :heavy_check_mark: | :heavy_multiplication_x: | [Code](https://github.com/manuelruder/artistic-videos) |
| 2016 | [Perceptual losses for real-time style transfer and super-resolution](https://arxiv.org/abs/1603.08155) | :heavy_multiplication_x: | :heavy_multiplication_x: | [Code](https://github.com/jcjohnson/neural-style) |
| 2016 | [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629) | :heavy_multiplication_x: | :heavy_check_mark: | :heavy_multiplication_x: |
| 2017 | [Real-time neural style transfer for videos](http://forestlinma.com/welcome_files/hzhuang_CVPR_2017.pdf) | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: |
| 2017 | [Artistic style transfer for videos and spherical images](https://arxiv.org/abs/1708.04538) | :heavy_check_mark: | :heavy_multiplication_x: | [Code](https://github.com/manuelruder/fast-artistic-videos) |
| 2017 | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) | :heavy_check_mark: | :heavy_multiplication_x: | [Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| 2018 | [Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) | :heavy_multiplication_x: | :heavy_check_mark: | [Code](https://github.com/yunjey/stargan) |
| 2019 | [Mocycle-GAN: Unpaired Video-to-Video Translation](https://arxiv.org/abs/1908.09514) | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: |
| 2019 | [Preserving Semantic and Temporal Consistency for Unpaired Video-to-Video Translation](https://arxiv.org/abs/1908.07683) | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: |
| 2020 | [StarGAN v2: Diverse Image Synthesis for Multiple Domains](https://arxiv.org/abs/1912.01865) | :heavy_multiplication_x: | :heavy_check_mark: | [Code](https://github.com/clovaai/stargan-v2) |

## Datasets:
* [FlyingChairs2](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)

## Requirements:
Mandatory:
* Python (tested with 3.6)
* CUDA capable device with
  - cudatoolkit (tested with 10.0.130)
  - cudnn (tested with 7.1)
* PyTorch (tested with 1.2.0)
* numpy, imageio

## Test System Info
* OS: Win10 (1909)
* CPU: AMD R5 3600
* RAM: 16GB DDR4 3200MHz CL14
* GPU: NVIDIA RTX 2080

Tested on Win10 and Ubuntu 16.04