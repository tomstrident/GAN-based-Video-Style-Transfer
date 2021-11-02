# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:35:37 2020

@author: Tom
"""

from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random

import numpy as np
#from skimage import io
#from flowlib import read

class DatasetFC2(data.Dataset):
  def __init__(self, dset_dir, sid, transform):
    self.data_dir1 = dset_dir + "styled-files/style0/"
    self.data_dir2 = dset_dir + "styled-files3/style0/"
    self.style_dir1 = dset_dir + "styled-files/style" + str(sid) + "/"
    self.style_dir2 = dset_dir + "styled-files3/style" + str(sid) + "/"
    
    self.transform = transform
    self.dataset = []
    
    self.preprocess()
    self.num_images = len(self.dataset)

  def __getitem__(self, index):
    img_id1, img_id2 = self.dataset[index]
    
    img1 = self.transform(Image.open(os.path.join(self.data_dir1, img_id1)))
    img2 = self.transform(Image.open(os.path.join(self.data_dir2, img_id2)))
    
    simg1 = self.transform(Image.open(os.path.join(self.style_dir1, img_id1)))
    simg2 = self.transform(Image.open(os.path.join(self.style_dir2, img_id2)))

    return img1, img2, simg1, simg2
  
  def __len__(self):
    return self.num_images
  
  def preprocess(self):
    data_list1 = os.listdir(self.data_dir1)
    data_list2 = os.listdir(self.data_dir2)
    
    data_list1.sort()
    data_list2.sort()
    
    assert len(data_list1) == len(data_list2)
    #assert len(data_list1) == len(style_dir1)
    #assert len(data_list1) == len(style_dir2)
    
    for filename in data_list1:
      filename2 = filename[:-4] + "_2" + filename[-4:]
      #print(filename2)
      #blah
      self.dataset.append([filename, filename2])

    random.seed(1234)
    random.shuffle(self.dataset)
    
    print('Finished preprocessing the dataset...')
  
class FC2_DatasetDataLoader():
  def __init__(self, opt, dset_dir="/home/tomstrident/datasets/FC2/", sid=1):
    transform = []
    #if mode == "train":
    #  transform.append(T.RandomHorizontalFlip())
    #transform.append(T.CenterCrop(crop_size))
    #transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    
    self.opt = opt
    self.dataset = DatasetFC2(dset_dir, sid, transform)
    
    print("dataset [%s] was created" % type(self.dataset).__name__)
    self.dataloader = torch.utils.data.DataLoader(
        self.dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads))

  def load_data(self):
    return self

  def __len__(self):
    """Return the number of data in the dataset"""
    return min(len(self.dataset), self.opt.max_dataset_size)

  def __iter__(self):
    """Return a batch of data"""
    for i, d in enumerate(self.dataloader):
      if i * self.opt.batch_size >= self.opt.max_dataset_size:
        break
      yield d
  