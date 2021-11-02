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
  def __init__(self, data_dir, style_dir, transform):
    self.data_dir = data_dir
    self.style_dir = style_dir
    self.transform = transform
    self.dataset = []
    self.attr2idx = {}
    self.idx2attr = {}
    
    self.preprocess()
    self.num_images = len(self.dataset)

  def __getitem__(self, index):
    img_id, simg_id, slbl = self.dataset[index]
    simg = self.transform(Image.open(os.path.join(self.style_dir, simg_id)))
    
    img_id = img_id[:-4] + ".npy"
    np_data = np.load(self.data_dir + img_id)[0] 
    
    img1 = self.transform(Image.fromarray(np.uint8(np_data[:,:,:3]*255.0)))
    img2 = self.transform(Image.fromarray(np.uint8(np_data[:,:,3:6]*255.0)))
    mask = torch.from_numpy(np.moveaxis(np_data[:,:,6:7], 2, 0))
    flow = torch.from_numpy(np.moveaxis(np_data[:,:,7:9], 2, 0))

    return img1, img2, simg, torch.FloatTensor(slbl), mask, flow
  
  def __len__(self):
    return self.num_images
  
  def preprocess(self):
    all_attr_names = os.listdir(self.style_dir)
    print(all_attr_names)
    all_attr_names.sort()
    all_attr_names = [all_attr_names[0]]
    print(all_attr_names)
    
    for i, attr_name in enumerate(all_attr_names):
      self.attr2idx[attr_name] = i
      self.idx2attr[i] = attr_name
      
      for filename in os.listdir(self.style_dir + attr_name):
        label = []
        
        for sub_attr_name in (["style0"] + all_attr_names):
          label.append(attr_name == sub_attr_name)

        #self.dataset.append([filename, attr_name + '\\' + filename, label])
        self.dataset.append([filename, attr_name + '/' + filename, label])
    
    random.seed(1234)
    random.shuffle(self.dataset)
    
    print('Finished preprocessing the style dataset...')
  
class FC2_DatasetDataLoader():
  def __init__(self, opt):
    transform = []
    #if mode == "train":
    #  transform.append(T.RandomHorizontalFlip())
    #transform.append(T.CenterCrop(crop_size))
    #transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    
    self.opt = opt
    #image_dir = "F:\\Datasets\\FC2\\DATAFiles\\"
    #style_dir = "F:\\Datasets\\FC2\\styled-files\\"
    self.dataset = DatasetFC2(opt.image_dir, opt.style_dir, transform)
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
  