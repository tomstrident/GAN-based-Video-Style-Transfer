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
    return out

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


class ConvTanh(ConvLayer):
  def __init__(self, in_channels, out_channels, kernel_size, stride):
    super(ConvTanh, self).__init__(in_channels, out_channels, kernel_size, stride)
    self.tanh = torch.nn.Tanh()

  def forward(self, x):
    out = super(ConvTanh, self).forward(x)
    return self.tanh(out/255) * 150 + 255/2

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.InstanceNorm2d(num_features, affine=True)
    #self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.embed = nn.Embedding(num_classes, num_features * 2)
    self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y):
    out = self.bn(x)
    '''
    y1 = torch.floor(y)
    y2 = torch.ceil(y)
    gamma1, beta1 = self.embed(y1.long()).chunk(2)
    gamma2, beta2 = self.embed(y2.long()).chunk(2)
    alpha2 = y - y1
    alpha1 = 1 - alpha2
    gamma = alpha1*gamma1 + alpha2*gamma2
    beta = alpha1*beta1 + alpha2*beta2
    '''
    
    gamma, beta = self.embed(y.long()).chunk(2)
    out = gamma.view(-1, self.num_features, 1, 1)*out + beta.view(-1, self.num_features, 1, 1)
    return out

class ConvInstRelu(ConvLayer):
  def __init__(self, in_channels, out_channels, kernel_size, stride, n_styles):
    super(ConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride)
    self.n_styles = n_styles

    if self.n_styles == 1:
      #print('ConvInstRelu InstanceNorm2d')
      self.instance = torch.nn.InstanceNorm2d(out_channels, affine=True)
    else:
      #print('ConvInstRelu ConditionalBatchNorm2d')
      self.instance = ConditionalBatchNorm2d(out_channels, self.n_styles)
      
    self.relu = torch.nn.ReLU()

  def forward(self, x, style_id):
    out = super(ConvInstRelu, self).forward(x)

    if self.n_styles == 1:
      out = self.instance(out)
    else:
      out = self.instance(out, style_id)
      
    out = self.relu(out)
    return out


class UpsampleConvLayer(torch.nn.Module):
  """Upsamples the input and then does a convolution.
  This method gives better results compared to ConvTranspose2d.
  ref: http://distill.pub/2016/deconv-checkerboard/
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
    super(UpsampleConvLayer, self).__init__()
    self.upsample = upsample
    reflection_padding = int(np.floor(kernel_size / 2))
    self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

  def forward(self, x):
    x_in = x
    if self.upsample:
      x_in = torch.nn.functional.interpolate(x_in, scale_factor=self.upsample)
    out = self.reflection_pad(x_in)
    out = self.conv2d(out)
    return out


class UpsampleConvInstRelu(UpsampleConvLayer):
  def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, n_styles=1):
    super(UpsampleConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride, upsample)
    self.n_styles = n_styles

    if self.n_styles == 1:
      #print('UpsampleConvInstRelu InstanceNorm2d')
      self.instance = torch.nn.InstanceNorm2d(out_channels, affine=True)
    else:
      #print('UpsampleConvInstRelu ConditionalBatchNorm2d')
      self.instance = ConditionalBatchNorm2d(out_channels, self.n_styles)
      
    self.relu = torch.nn.ReLU()

  def forward(self, x, style_id):
    out = super(UpsampleConvInstRelu, self).forward(x)

    if self.n_styles == 1:
      out = self.instance(out)
    else:
      out = self.instance(out, style_id)
      
    out = self.relu(out)
    return out

class ResidualBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, n_styles=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
    
    self.in1 = torch.nn.InstanceNorm2d(out_channels, affine=True)
    self.in2 = torch.nn.InstanceNorm2d(out_channels, affine=True)
    
    '''
    if self.n_styles == 1:
      self.in1 = torch.nn.InstanceNorm2d(out_channels, affine=True)
      self.in2 = torch.nn.InstanceNorm2d(out_channels, affine=True)
      #print('ResidualBlock InstanceNorm2d')
    else:
      self.in1 = ConditionalBatchNorm2d(out_channels, num_styles)
      self.in2 = ConditionalBatchNorm2d(out_channels, num_styles)
      #print('ResidualBlock ConditionalBatchNorm2d')
    '''
      
    self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

    self.relu = torch.nn.ReLU()
    self.layer_strength = torch.nn.Parameter(torch.tensor([1],dtype=torch.float32,requires_grad=True))

  def forward(self, x, style_strength, s_id):
    strength = style_strength*self.layer_strength
    strength = 2*strength.abs()/(1+strength.abs())  
    residual = x
    
    out = self.relu(self.in1(self.conv1(x)))
    out = self.in2(self.conv2(out))
    
    '''
    if self.n_styles == 1:
      out = self.relu(self.in1(self.conv1(x)))
      out = self.in2(self.conv2(out))
    else:
      out = self.relu(self.in1(self.conv1(x), style_id))
      out = self.in2(self.conv2(out), style_id)
    '''

    out = strength*out + residual
    return out

class FastStyleNet(SelectiveLoadModule):
  def __init__(self, num_inp, n_styles=1):
    super(FastStyleNet, self).__init__()

    self.conv1 = ConvInstRelu(num_inp, 32, kernel_size=9, stride=1, n_styles=n_styles)
    self.conv2 = ConvInstRelu(32, 64, kernel_size=3, stride=2, n_styles=n_styles)
    self.conv3 = ConvInstRelu(64, 128, kernel_size=3, stride=2, n_styles=n_styles)

    self.res1 = ResidualBlock(128, 128, n_styles=n_styles)
    self.res2 = ResidualBlock(128, 128, n_styles=n_styles)
    self.res3 = ResidualBlock(128, 128, n_styles=n_styles)
    self.res4 = ResidualBlock(128, 128, n_styles=n_styles)
    self.res5 = ResidualBlock(128, 128, n_styles=n_styles)

    self.deconv1 = UpsampleConvInstRelu(128, 64, kernel_size=3, stride=1, upsample=2, n_styles=n_styles)
    self.deconv2 = UpsampleConvInstRelu(64, 32, kernel_size=3, stride=1, upsample=2, n_styles=n_styles)
    self.deconv3 = ConvTanh(32, 3, kernel_size=9, stride=1)

  def forward(self, x, style_strength=1.0, s_id=0):
    x = self.conv1(x, s_id)
    x = self.conv2(x, s_id)
    x = self.conv3(x, s_id)

    x = self.res1(x, style_strength, s_id)
    x = self.res2(x, style_strength, s_id)
    x = self.res3(x, style_strength, s_id)
    x = self.res4(x, style_strength, s_id)
    x = self.res5(x, style_strength, s_id)

    features = x

    x = self.deconv1(x, s_id)
    x = self.deconv2(x, s_id)
    x = self.deconv3(x)

    return (features, x)
