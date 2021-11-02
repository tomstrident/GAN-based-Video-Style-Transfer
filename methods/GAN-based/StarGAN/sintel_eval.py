# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:34:30 2021

@author: Tom
"""

import sys
sys.path.append('raft')

import torch
from torch.utils import data
from torchvision import transforms
from utils.utils import InputPadder
import torchvision.utils as vutils

import os
import json
import time
import random
import numpy as np
from PIL import Image

from collections import OrderedDict
from tqdm import tqdm

from raft.raft import RAFT
from flowtools import fbcCheckTorch, warp

#from models import create_model

def denormalize(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)

def save_image(x, ncol, filename):
  x = denormalize(x)
  vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

def save_json(json_file, filename):
  with open(filename, 'w') as f:
    json.dump(json_file, f, indent=4, sort_keys=False)

def initRaftModel(opt, device):
  model = torch.nn.DataParallel(RAFT(opt))
  model.load_state_dict(torch.load("raft/models/raft-chairs.pth"))

  model = model.module
  model.to(device)
  model.eval()
  
  return model

def computeRAFT(net, img1, img2, it=20):
  B, C, H, W = img1.size()
  with torch.no_grad():
    padder = InputPadder(img1.shape)
    image1, image2 = padder.pad(img1, img2)
    flow_low, flow_up = net(image1, image2, iters=it, test_mode=True)
    
  return flow_up[:,:,:H,:]

class SingleSintelVideo(data.Dataset):
  def __init__(self, vid_dir, transform, lt_len=5):
    self.vid_dir = vid_dir
    
    self.transform = transform
    self.dataset = []
    self.lt_len = lt_len
    
    self.preprocess()
    self.num_images = len(self.dataset)
    #print("SingleSintelVideo", self.num_images)

  def __getitem__(self, index):
    fid = self.dataset[index]
    img =  self.transform(Image.open(fid))
    last_img = torch.Tensor([0])[0].type(torch.LongTensor)
    past_img = torch.Tensor([0])[0].type(torch.LongTensor)
  
    if index > 0:
      last_fid = self.dataset[index - 1]
      last_img =  self.transform(Image.open(last_fid))
      
    if index >= self.lt_len:
      past_fid = self.dataset[index - self.lt_len]
      past_img =  self.transform(Image.open(past_fid))
    
    return img, last_img, past_img
  
  def __len__(self):
    return self.num_images
  
  def preprocess(self):
    frame_list = os.listdir(self.vid_dir)
    frame_list.sort()
    
    for fid in frame_list:
      self.dataset.append(os.path.join(self.vid_dir, fid))

    #print("Dataset Len:", len(self.dataset))
    
    random.seed(1234)

def computeTCL(net, model, img_fake, img1, img2, c_trg):
  ff_last = computeRAFT(model, img2, img1)
  bf_last = computeRAFT(model, img1, img2)
  mask_last = fbcCheckTorch(ff_last, bf_last)
  #warp_last = warp(net.generator(img2, s_trg), bf_last)
  warp_last = warp(net(img2, c_trg), bf_last)
  return ((mask_last*(img_fake - warp_last))**2).mean()**0.5

def save_dict_as_json(out_id, data_dict, out_path, num_domains):
  dict_mean = 0
  dict_mean_s = np.zeros(num_domains - 1)
  
  for key, value in data_dict.items():
    len_3 = len(data_dict)/3
    dict_mean += value / len(data_dict)
    
    for d in range(1, num_domains):
      if ("_s" + str(d)) in key:
        dict_mean_s[d-1] += value / len_3
        
  data_dict[out_id + "_mean"] = float(dict_mean)
  
  for d in range(1, num_domains):
    data_dict[out_id + "_mean_s" + str(d)] = float(dict_mean_s[d-1])
    
  filename = os.path.join(out_path, out_id + '.json')
  save_json(data_dict, filename)

'''
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='models/raft-chairs.pth', help="restore checkpoint")
parser.add_argument('--path', default='demo-frames', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()
'''

def evaluate_sintel(args, sintel_dir="D:/Datasets/MPI-Sintel-complete/"):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  out_path = "G:/Code/ConGAN/eval/"
  raft_model = initRaftModel(args, device)
  
  #domains = os.listdir(args.style_dir)
  #domains.sort()
  num_domains = 4#len(domains)
  
  transform = []
  transform.append(transforms.ToTensor())
  transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
  transform = transforms.Compose(transform)
  
  train_dir = os.path.join(sintel_dir, "training", "final")
  train_list = os.listdir(train_dir)
  train_list.sort()
  
  test_dir = os.path.join(sintel_dir, "test", "final")
  test_list = os.listdir(test_dir)
  test_list.sort()
  
  video_list = [os.path.join(train_dir, vid) for vid in train_list]
  video_list += [os.path.join(test_dir, vid) for vid in test_list]
  
  vid_list = train_list + test_list
  tcl_st_dict = {}
  tcl_lt_dict = {}
  
  tcl_st_dict = OrderedDict()
  tcl_lt_dict = OrderedDict()
  dt_dict = OrderedDict()
  
  args.checkpoints_dir = os.getcwd() + "\\checkpoints\\"
  model_list = os.listdir(args.checkpoints_dir)
  model_list.sort()
  
  for j, vid_dir in enumerate(video_list):
    vid = vid_list[j]

    sintel_dset = SingleSintelVideo(vid_dir, transform)
    loader = data.DataLoader(dataset=sintel_dset, batch_size=1, shuffle=False, num_workers=0)
    
    for y in range(1, num_domains):
      #y_trg = torch.Tensor([y])[0].type(torch.LongTensor).to(device)
      key = vid + "_s" + str(y)
      vid_path = os.path.join(out_path, key)
      if not os.path.exists(vid_path):
        os.makedirs(vid_path)
    
      tcl_st_vals = []
      tcl_lt_vals = []
      dt_vals = []

      args.name = model_list[y-1]
      #args.model = "cycle_gan"
      model = create_model(args)
      model.setup(args)
      
      for i, imgs in enumerate(tqdm(loader, total=len(loader))):
        img, img_last, img_past = imgs
        
        img = img.to(device)
        img_last = img_last.to(device)
        img_past  = img_past.to(device)
        
        t_start = time.time()
        x_fake = model.forward_eval(img)
        t_end = time.time()
        
        dt_vals.append((t_end - t_start)*1000)
        
        if i > 0:
          tcl_st = computeTCL(model, raft_model, x_fake, img, img_last)
          tcl_st_vals.append(tcl_st.cpu().numpy())
        
        if i >= 5:
          tcl_lt = computeTCL(model, raft_model, x_fake, img, img_past)
          tcl_lt_vals.append(tcl_lt.cpu().numpy())
          
        filename = os.path.join(vid_path, "frame_%04d.png" % i)
        save_image(x_fake[0], ncol=1, filename=filename)
        
      tcl_st_dict["TCL-ST_" + key] = float(np.array(tcl_st_vals).mean())
      tcl_lt_dict["TCL-LT_" + key] = float(np.array(tcl_lt_vals).mean())
      dt_dict["DT_" + key] = float(np.array(dt_vals).mean())
    
  save_dict_as_json("TCL-ST", tcl_st_dict, out_path, num_domains)
  save_dict_as_json("TCL-LT", tcl_lt_dict, out_path, num_domains)
  save_dict_as_json("DT", dt_dict, out_path, num_domains)


  