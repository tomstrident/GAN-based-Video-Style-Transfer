# GAN-based-Video-Style-Transfer
Source files of my master thesis at TU Graz [GAN-based-Video-Style-Transfer](https://drive.google.com/file/d/1O2scchvoUWZtw6c60DvVeOOd4fiZLOBQ/view?usp=sharing).

## Methods:
In this work various optimization-based, learning-based and GAN-based style transfer methods and deviations have been evaluated and compared.

| Year | Type | Paper | Temporal | Multi-Domain | Code |
| ---- | ---- | ----- | -------- | ------------ | ---- |
| 2015 | obst | []() | []() | ---- |

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

* Optimization-based methods:
  - Gatys et al. [](https://github.com/leongatys/PytorchNeuralStyleTransfer)
  - Ruder et al. [](https://github.com/manuelruder/artistic-videos)
* Learning-based methods:
  - Johnson et al. [](https://github.com/jcjohnson/neural-style)
  - Dumoulin et al. []()
  - Huang et al. []()
  - Ruder et al. [](https://github.com/manuelruder/fast-artistic-videos)
* GAN-based methods:
  - Zhu et al. [](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  - Choi et al. [](https://github.com/yunjey/stargan)
  - Chen et al. []()
  - Park et al. []()
  - Choi et al. [](https://github.com/clovaai/stargan-v2)

## Datasets:
[FlyingChairs2](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
[Sintel](http://sintel.is.tue.mpg.de/)

## Requirements:
Mandatory:
* Python (tested with 3.6)
* CUDA capable device with
  - cudatoolkit (tested with 10.0.130)
  - cudnn (tested with 7.1)
* PyTorch (tested with 1.2.0)
* numpy, imageio, OpenCV3, scikit-image, PyQt5

## Install:
Just install all mandatory packages from the requirements list. Everything else should be handled by Python. But be sure to adjust all paths when training or testing.

## Training:

## Testing:

## Demo:
The demo file `demo.py` includes an example for training and testing. Be sure to adjust the standard paths of `train_net` and `infer_test` and the batch size or input resolution if GPU memory is restricted.

## Test System Info
* OS: Win10 (1909)
* CPU: AMD R5 3600
* RAM: 16GB DDR4 3200MHz CL14
* GPU: NVIDIA RTX 2080

Tested on Win10 and Ubuntu 16.04