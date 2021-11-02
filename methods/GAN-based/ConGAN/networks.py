# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16, vgg19
from collections import namedtuple


# From https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
class Vgg16(torch.nn.Module):
  def __init__(self, device='cpu'):
    super(Vgg16, self).__init__()
    vgg_pretrained_features = vgg16(pretrained=True).features
    
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    
    for x in range(4):
      self.slice1.add_module(str(x), vgg_pretrained_features[x].to(device))
    for x in range(4, 9):
      self.slice2.add_module(str(x), vgg_pretrained_features[x].to(device))
    for x in range(9, 16):
      self.slice3.add_module(str(x), vgg_pretrained_features[x].to(device))
    for x in range(16, 23):
      self.slice4.add_module(str(x), vgg_pretrained_features[x].to(device))
    
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, X):
    h = self.slice1(X)
    h_relu1_2 = h
    h = self.slice2(h)
    h_relu2_2 = h
    h = self.slice3(h)
    h_relu3_3 = h
    h = self.slice4(h)
    h_relu4_3 = h
    vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
    out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
    return out

class Vgg19(torch.nn.Module):
  def __init__(self, device='cpu'):
    super(Vgg19, self).__init__()
    vgg_pretrained_features = vgg19(pretrained=True).features
    
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    self.slice5 = torch.nn.Sequential()
    
    self.instance = torch.nn.InstanceNorm2d(512, affine=True)
    
    for x in range(2):
      self.slice1.add_module(str(x), vgg_pretrained_features[x].to(device))
    for x in range(2, 7):
      self.slice2.add_module(str(x), vgg_pretrained_features[x].to(device))
    for x in range(7, 12):
      self.slice3.add_module(str(x), vgg_pretrained_features[x].to(device))
    for x in range(12, 21):
      self.slice4.add_module(str(x), vgg_pretrained_features[x].to(device))
    for x in range(21, 30):
      self.slice5.add_module(str(x), vgg_pretrained_features[x].to(device))
    
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, X):
    h_relu1 = self.slice1(X)
    h_relu2 = self.slice2(h_relu1)
    h_relu3 = self.slice3(h_relu2)
    h_relu4 = self.slice4(h_relu3)
    h_relu5 = self.slice5(h_relu4)
    vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
    out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
    return self.instance(out[-1])
    
# Rest of the file based on https://github.com/irsisyphus/reconet

class SelectiveLoadModule(torch.nn.Module):
  """Only load layers in trained models with the same name."""
  def __init__(self):
    super(SelectiveLoadModule, self).__init__()

  def forward(self, x):
    return x

  def load_state_dict(self, state_dict):
    """Override the function to ignore redundant weights."""
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name in own_state:
        own_state[name].copy_(param)

class ConvLayer(torch.nn.Module):
  """Reflection padded convolution layer."""
  def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
    super(ConvLayer, self).__init__()
    reflection_padding = int(np.floor(kernel_size / 2))
    self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

  def forward(self, x):
    out = self.reflection_pad(x)
    out = self.conv2d(out)
    return out

class ConvInstRelu(ConvLayer):
  def __init__(self, in_channels, out_channels, kernel_size, stride):
    super(ConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride)

    self.instance = torch.nn.InstanceNorm2d(out_channels, affine=True)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    out = super(ConvInstRelu, self).forward(x)
    out = self.instance(out)
    out = self.relu(out)
    return out
    
class ConvSig(ConvLayer):
  def __init__(self, in_channels, out_channels, kernel_size, stride):
    super(ConvSig, self).__init__(in_channels, out_channels, kernel_size, stride)

    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    out = super(ConvSig, self).forward(x)
    out = self.sigmoid(out)
    return out

class FusionBlock(SelectiveLoadModule):
  def __init__(self):
    super(FusionBlock, self).__init__()

    self.conv1 = ConvInstRelu(3, 32, kernel_size=3, stride=1)
    self.conv2 = ConvInstRelu(32, 32, kernel_size=3, stride=1)
    self.conv3 = ConvSig(32, 1, kernel_size=3, stride=1)

  def forward(self, simg, wimg):
    m = self.conv1(simg - wimg)
    m = self.conv2(m)
    m = self.conv3(m)

    return m*wimg + (1 - m)*simg

from PIL import Image
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def main():
  block = FusionBlock().to(DEVICE)
  
  img = load_image("img.png")
  img2 = load_image("img2.png")
  
  test = block(img, img2)
  print(test.shape)

if __name__ == '__main__':
  main()
