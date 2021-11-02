# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from flowlib import read
from skimage import io

device='cuda'

class FlyingChairs2Dataset(Dataset):
  
  def __init__(self, path, batch_size, transform=None):
    self.path = path
    
    print('loading FlyingChairs2 ...')
    
    #self.init_path = path + 'INITFiles/'
    self.data_path = path + 'DATAFiles/'
    
    #self.init_list = os.listdir(self.init_path)
    self.data_list = os.listdir(self.data_path)
    dset_size = len(self.data_list)
    
    assert dset_size == 22232
    #dset_size = 9627
    
    self.length = np.int32(np.floor(dset_size / batch_size)*batch_size)
    
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    batch_data = np.load(self.data_path + self.data_list[idx])
    #print(batch_data.shape)
    
    #img1 = batch_data[0,:,:,0:3]
    #img2 = batch_data[0,:,:,3:6]
    #mask = batch_data[0,:,:,6:7]
    #flow = batch_data[0,:,:,7:9]
    #init = batch_data[0,:,:,9:12]*255.0???
    
    #imgs = batch_data[0,:,:,0:6]
    #img2 = batch_data[0,:,:,3:6]
    #img3 = batch_data[0,:,:,6:9]
    #mask = batch_data[0,:,:,9:10]
    #mask2 = batch_data[0,:,:,10:11]
    #flow = batch_data[0,:,:,11:13]
    #flow2 = batch_data[0,:,:,13:15]
    
    imgs = batch_data[0,:,:,0:6]
    mask = batch_data[0,:,:,6:7]
    flow = batch_data[0,:,:,7:9]
    
    img_size = (384, 512)
    batch_size = (256, 256)
    h, w = (np.array(batch_size)/2).astype(np.int32)
    ih, iw = (np.array(img_size)/2).astype(np.int32)
    
    #num = toString7(idx)
    #init = []#io.imread(self.path+"train/"+num+"-sty_0.png")[ih-h:ih+h,iw-w:iw+w,:]/255.0
    
    #print(imgs.shape)
    #print(mask.shape)
    #print(flow.shape)
    #print(init.shape)
    #blah
    
    #print(imgs.shape)
    #print(mask.shape)
    #print(flow.shape)
    #blah
    
    imgs = torch.from_numpy(imgs).to(device).permute(2, 0, 1).float()
    mask = torch.from_numpy(mask).to(device).permute(2, 0, 1).float()
    flow = torch.from_numpy(flow).to(device).permute(2, 0, 1).float()
    #init = torch.from_numpy(init).to(device).permute(2, 0, 1).float()
    
    return (imgs, mask, flow)
    #return (imgs, [], [], [])

class Hollywood2Dataset(Dataset):
  def __init__(self, path, batch_size, transform=None):
    self.path = path
    
    print('loading Hollywood2 ...')
    
    self.data_path = path + 'DATAFiles/'
    self.data_list = os.listdir(self.data_path)
    dset_size = len(self.data_list)
    
    assert dset_size == 9627
    
    self.length = np.int32(np.floor(dset_size / batch_size)*batch_size)
    
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    frames, flows, masks = np.load(self.data_path + self.data_list[idx], allow_pickle=True)
    
    frames = np.concatenate(frames, axis=2)
    flows = np.concatenate(flows, axis=2)
    masks = np.concatenate(masks, axis=2)

    imgs = torch.from_numpy(frames).to(device).permute(2, 0, 1).float()
    masks = torch.from_numpy(masks).to(device).permute(2, 0, 1).float()
    flows = torch.from_numpy(flows).to(device).permute(2, 0, 1).float()
    
    return (imgs, masks, flows)
  
class COCODataset(Dataset):
  def __init__(self, path, batch_size, transform=None):
    self.path = path
    
    print('loading COCO ...')
    
    self.data_path = path + 'DATAFiles/'
    self.data_list = os.listdir(self.data_path)
    dset_size = len(self.data_list)
    
    assert dset_size == 9627
    
    self.length = np.int32(np.floor(dset_size / batch_size)*batch_size)
    
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    frames, flows, masks = np.load(self.data_path + self.data_list[idx], allow_pickle=True)
    
    frames = np.concatenate(frames, axis=2)
    flows = np.concatenate(flows, axis=2)
    masks = np.concatenate(masks, axis=2)

    imgs = torch.from_numpy(frames).to(device).permute(2, 0, 1).float()
    masks = torch.from_numpy(masks).to(device).permute(2, 0, 1).float()
    flows = torch.from_numpy(flows).to(device).permute(2, 0, 1).float()
    
    return (imgs, masks, flows)

class SintelDataset(Dataset):
  def __init__(self, path, video_id, transform=None):
    self.path = path
    
    print('loading ' + video_id + ' ...')
    
    if not os.path.exists(path):
      print('Invalid path!')
      assert False
      
    sintel_path = path + 'MPI-Sintel-complete/training/'
    self.fc5_path = "F:/Datasets/Sintel5/" + video_id + "/"

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
    
    self.length = len(self.frames_list)
    
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    frame = io.imread(self.frames_path + self.frames_list[idx])/255.0
    
    if idx == 0:
      flow = np.zeros(frame.shape[:2] + (2,))
      mask = np.zeros(frame.shape[:2] + (1,))
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
      
      lt_flow = torch.from_numpy(lt_flow).to(device).permute(2, 0, 1).float()
      lt_mask = torch.from_numpy(lt_mask).to(device).permute(2, 0, 1).float()
      #print(lt_flow.shape)
      #print(lt_mask.shape)
      
      #blah

    #print(frame.shape)
    #print(flow.shape)
    #print(mask.shape)

    frame = torch.from_numpy(frame).to(device).permute(2, 0, 1).float()
    flow = torch.from_numpy(flow).to(device).permute(2, 0, 1).float()
    mask = torch.from_numpy(mask).to(device).permute(2, 0, 1).float()
    
    return (frame, mask, flow, [lt_flow, lt_mask])
    
class CombinedDataset(Dataset):

  def __init__(self, 
               fc2_path='F:/Datasets/FC2/', 
               co2_path='F:/Datasets/CO2/', 
               hw2_path='F:/Datasets/HW2/', 
               batch_size=6, transform=None):
    self.fc2=FlyingChairs2Dataset(fc2_path, batch_size)
    self.co2=COCODataset(co2_path, batch_size)
    self.hw2=Hollywood2Dataset(hw2_path, batch_size)
    
  def __len__(self):
    return len(self.fc2) + len(self.co2) + len(self.hw2)

  def __getitem__(self, idx):
    if(idx < len(self.fc2)):
      return self.fc2.__getitem__(idx)
    elif idx < len(self.fc2) + len(self.co2):
      return self.co2.__getitem__(idx - len(self.fc2))
    else:
      return self.hw2.__getitem__(idx - len(self.fc2) - len(self.co2))
    
class ChairsSDHomDataset(Dataset):
  
  def __init__(self, path, batch_size, transform=None):
    self.path = path
    
    print('loading ChairsSDHom ...')
    self.data_path = path
    self.data_list = os.listdir(self.data_path)
    dset_size = 20966
    self.length = np.int32(np.floor(dset_size / batch_size)*batch_size)
    
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    batch_data = np.load(self.data_path + self.data_list[idx], allow_pickle=True)
    
    imgs = batch_data[:,:,0:6]
    flow = batch_data[:,:,6:8]
    mask = batch_data[:,:,8:9]
    
    img_size = (384, 512)
    batch_size = (256, 256)
    h, w = (np.array(batch_size)/2).astype(np.int32)
    ih, iw = (np.array(img_size)/2).astype(np.int32)
    
    imgs = torch.from_numpy(imgs[ih-h:ih+h,iw-w:iw+w,:]).to(device).permute(2, 0, 1).float()
    flow = torch.from_numpy(flow[ih-h:ih+h,iw-w:iw+w,:]).to(device).permute(2, 0, 1).float()
    mask = torch.from_numpy(mask[ih-h:ih+h,iw-w:iw+w,:]).to(device).permute(2, 0, 1).float()
    
    return (imgs, mask, flow)
  
  