# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:03:04 2020

@author: Tom
"""

import os
import torch

import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class GatysDataset(data.Dataset):
  def __init__(self, image_dir):
    self.transform = T.Compose([T.Scale(512),
                                T.ToTensor(),
                                T.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                                T.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                                T.Lambda(lambda x: x.mul_(255))])
    
    self.image_dir = image_dir
    self.dataset = os.listdir(image_dir)
    self.dataset.sort()
    self.num_images = len(self.dataset)

  def __getitem__(self, index):
    filename = self.dataset[index]
    image = Image.open(self.image_dir + filename)
    image = self.transform(image)
    #image = Variable(image.unsqueeze(0).cuda())
    return image, filename
  
  def __len__(self):
    return self.num_images

class FlyingChairs2(data.Dataset):
  def __init__(self, image_dir):
    self.transform = T.Compose([T.Scale(512),
                                T.ToTensor(),
                                T.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                                T.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                                T.Lambda(lambda x: x.mul_(255))])
    
    self.image_dir = image_dir
    self.dataset = os.listdir(image_dir)
    self.dataset.sort()
    self.num_images = len(self.dataset)

  def __getitem__(self, index):
    filename = self.dataset[index]
    np_data = np.load(self.image_dir + filename)[0]
    #image = Image.fromarray(np.uint8(np_data[:,:,0:3]*255))
    image = Image.fromarray(np.uint8(np_data[:,:,3:6]*255))
    image = self.transform(image)
    #filename = filename[:-4] + ".jpg"
    filename = filename[:-4] + "_2.jpg"
    return image, filename
  
  def __len__(self):
    return self.num_images

def getGatysLoader(image_dir, batch_size=4, num_w=0):
  return data.DataLoader(dataset=GatysDataset(image_dir),
                         batch_size=batch_size, shuffle=False, num_workers=0)

def getFC2Loader(image_dir, batch_size=4, num_w=0):
  return data.DataLoader(dataset=FlyingChairs2(image_dir),
                         batch_size=batch_size, shuffle=False, num_workers=0)