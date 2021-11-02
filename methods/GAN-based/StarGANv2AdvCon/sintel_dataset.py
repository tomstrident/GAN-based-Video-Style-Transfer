# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:07:40 2020

@author: Tom
"""

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

import numpy as np
from skimage import io
from flowlib import read

# =============================================================================

class SintelDataset(data.Dataset):
  def __init__(self, path, video_id, transform):
    self.path = path
    
    print('loading ' + video_id + ' ...')
    
    if not os.path.exists(path):
      print('Invalid path!')
      assert False

    #sintel_path = path + 'MPI-Sintel-complete/training/'
    sintel_path = path + 'Sintel/'
    self.fc5_path = "/home/tomstrident/datasets/Sintel5/" + video_id + "/"

    self.frames_path = sintel_path + 'final/' + video_id + '/'
    self.flows_path = sintel_path + 'flow/' + video_id + '/'
    self.masks_path = sintel_path + 'occlusions/' + video_id + '/'
    
    self.frames_list = os.listdir(self.frames_path)
    self.flows_list = os.listdir(self.flows_path)
    self.masks_list = os.listdir(self.masks_path)
    self.lt_data_list = os.listdir(self.fc5_path)
    
    self.frames_list.sort(reverse=True)
    self.flows_list.sort(reverse=True)
    self.masks_list.sort(reverse=True)
    self.lt_data_list.sort(reverse=True)
    
    self.transform = transform
    self.length = len(self.frames_list)
    
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    #frame = io.imread(self.frames_path + self.frames_list[idx])/255.0
    frame = self.transform(Image.open(self.frames_path + self.frames_list[idx]))
    
    if idx == 0:
      flow = np.zeros(frame.shape[1:] + (2,))
      mask = np.zeros(frame.shape[1:] + (1,))
    else:
      flow = read(self.flows_path + self.flows_list[idx-1])
      mask = io.imread(self.masks_path + self.masks_list[idx-1])/255.0
      mask = 1.0 - mask.reshape(mask.shape + (1,))
    
    offset = 5
    if idx - offset < 0 or idx == self.length - 1:
      lt_flow = []
      lt_mask = []
    else:
      #print(self.frames_path + self.frames_list[idx])
      #print(self.fc5_path + self.lt_data_list[idx-offset])
      data = np.load(self.fc5_path + self.lt_data_list[idx-offset], allow_pickle=True)
      #print(data.shape)
      
      lt_flow = data[0,:,:,:2]
      lt_mask = data[0,:,:,2]
      lt_mask = lt_mask.reshape(lt_mask.shape + (1,))
      
      lt_flow = torch.from_numpy(lt_flow).to("cuda").permute(2, 0, 1).float()
      lt_mask = torch.from_numpy(lt_mask).to("cuda").permute(2, 0, 1).float()
      #print(lt_flow.shape)
      #print(lt_mask.shape)
      
      #blah

    #frame = torch.from_numpy(frame).to("cuda").permute(2, 0, 1).float()
    flow = torch.from_numpy(flow).to("cuda").permute(2, 0, 1).float()
    mask = torch.from_numpy(mask).to("cuda").permute(2, 0, 1).float()
    
    return (frame, mask, flow, [lt_flow, lt_mask])

def getTestDatasetLoader(path, video_id):
  transform = []
  transform.append(T.ToTensor())
  transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
  transform = T.Compose(transform)
  
  return data.DataLoader(dataset=SintelDataset(path, video_id, transform),
                                   batch_size=1, shuffle=False, num_workers=0)