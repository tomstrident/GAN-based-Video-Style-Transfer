# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:57:20 2020

@author: Tom
"""

import time
import os 
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict

from tqdm import tqdm

import random
import sys
sys.path.append('raft')
from raft.raft import RAFT
from flowtools import fbcCheckTorch, warp

from torch.utils import data
from utils.utils import InputPadder
import torchvision.utils as vutils
from torchvision import transforms as T

import json
import shutil
from sg2_core.data_loader import get_loaderFC2
from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
#from core.data_loader import get_eval_loader
from sg2_core import utils
import numpy as np

def denormalize(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)

def save_image(x, ncol, filename):
  #x = denormalize(x)
  vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

def save_json(json_file, filename):
  with open(filename, 'w') as f:
    json.dump(json_file, f, indent=4, sort_keys=False)

def initRaftModel(opt):
  model = torch.nn.DataParallel(RAFT(opt))
  model.load_state_dict(torch.load("raft/models/raft-chairs.pth"))

  model = model.module
  model.to('cuda')
  model.eval()
  
  return model

def computeRAFT(net, img1, img2, it=20):
  B, C, H, W = img1.size()
  with torch.no_grad():
    padder = InputPadder(img1.shape)
    image1, image2 = padder.pad(img1, img2)
    flow_low, flow_up = net(image1, image2, iters=it, test_mode=True)
    
  return flow_up[:,:,:H,:]

def create_task_folders(eval_dir, task):
  path_ref = os.path.join(eval_dir, task + "/ref")
  path_fake = os.path.join(eval_dir, task + "/fake")
  
  if os.path.exists(path_ref):
    #print("exists")
    shutil.rmtree(path_ref, ignore_errors=True)
  os.makedirs(path_ref)
    
  if os.path.exists(path_fake):
    #print("exists")
    shutil.rmtree(path_fake, ignore_errors=True)
  os.makedirs(path_fake)

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

def computeTCL(net, model, img_fake, img1, img2, sid):
  ff_last = computeRAFT(model, img2, img1)
  bf_last = computeRAFT(model, img1, img2)
  mask_last = fbcCheckTorch(ff_last, bf_last)
  warp_last = warp(net.run(img2, sid), bf_last)
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


#==============================================================================

#vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        #out['r52'] = F.relu(self.conv5_2(out['r51']))
        #out['r53'] = F.relu(self.conv5_3(out['r52']))
        #out['r54'] = F.relu(self.conv5_4(out['r53']))
        #out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
      
# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

class OBST():
  def __init__(self, batch_size=1):
    #pyr_shapes = [512, 256, 128, 64]
    #max_iters = [25, 50, 75]
    
    self.max_iters = [50, 40, 30]
    
    #get network
    self.vgg = VGG()
    model_dir = os.getcwd() + '/Models/'
    model_weights = torch.load(model_dir + 'vgg_conv.pth')
    del model_weights["conv5_2.weight"]
    del model_weights["conv5_2.bias"]
    del model_weights["conv5_3.weight"]
    del model_weights["conv5_3.bias"]
    del model_weights["conv5_4.weight"]
    del model_weights["conv5_4.bias"]
    
    self.vgg.load_state_dict(model_weights)
    for param in self.vgg.parameters():
      param.requires_grad = False
    if torch.cuda.is_available():
      self.vgg.cuda()
    
    #define layers, loss functions, weights and compute optimization targets
    #self.style_layers = ['r11','r21','r31','r41', 'r51']
    self.style_layers = ['r21','r31','r41']
    self.content_layers = ['r42']
    self.loss_layers = self.style_layers + self.content_layers
    self.loss_fns = [GramMSELoss()] * len(self.style_layers) + [nn.MSELoss()] * len(self.content_layers)
    if torch.cuda.is_available():
      self.loss_fns = [loss_fn.cuda() for loss_fn in self.loss_fns]
        
    #these are good weights settings:
    beta = 1e2 # 1e3
    style_weights = [beta/n**2 for n in [128,256,512]]
    #style_weights = [beta/n**2 for n in [64,128,256,512,512]]
    content_weights = [1e0]
    self.weights = style_weights + content_weights
    
    self.batch_size = batch_size
    
    self.style_img_dir = os.getcwd() + '/Images/'
    self.style_img_list = os.listdir(self.style_img_dir)
    self.style_img_list.sort()
    
    self.style_targets = []

  def postp(self, tensor): # to clip results in the range [0,1]
    t = self.postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = self.postpb(t)
    return img
      
  def postp2(self, tensor): # to clip results in the range [0,1]
    t = self.postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = self.postpb2(t)
    return img

  def set_shapes(self, pyr_shapes):
    #self.pyr_shapes = [(64, 64), (128, 128), (256, 256)]
    #self.pyr_shapes = [(109, 256), (218, 512), (436, 1024)]
    self.pyr_shapes = pyr_shapes
    
    self.preps = []
    for pyr_shape in self.pyr_shapes:
      print(pyr_shape)
      self.preps.append(transforms.Compose([transforms.Resize(pyr_shape),
                   transforms.ToTensor(),
                   transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                   transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),#subtract imagenet mean                              
                   transforms.Lambda(lambda x: x.mul_(255))]))
      
    
    self.postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                               transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                    std=[1,1,1]),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                               ])
    self.postpb2 = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize(pyr_shapes[-1]),
                                 transforms.Grayscale(num_output_channels=3)])
    self.postpb = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize(pyr_shapes[-1])])

  def set_style(self, sid):
    if not isinstance(sid, list):
      sid = [sid for _ in range(self.batch_size)]
    
    style_targets = []
    #for sid, style_img in enumerate(style_img_list):
    for prep in self.preps:
      style_image = [prep(Image.open(self.style_img_dir + self.style_img_list[sid[b]])).unsqueeze(0) for b in range(self.batch_size)]
      style_image = torch.cat(style_image, dim=0)
      style_image = Variable(style_image.cuda())
      #style_image = Variable(style_image.unsqueeze(0).cuda())
    
      #compute optimization targets
      style_targets.append([GramMatrix()(A).detach() for A in self.vgg(style_image,self.style_layers)])
      del style_image
      
    self.style_targets = style_targets

  def run(self, pre, img, sid, mask, weight_tcl=0):#2000
    content_image = img.to("cuda")
    
    # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
    opt_img = Variable(pre.data.clone(), requires_grad=True)
    
    #content_image = prep(Image.open(dset_img_dir + dset_img))
    #content_image = Variable(content_image.unsqueeze(0).cuda())
    #print(content_image.shape)
    
    wimg = pre
    mimg = mask.repeat(1, 3, 1, 1)
    
    content_targets = []
    warp_targets = []
    mask_targets = []
    for pyr_idx, pyr_shape in enumerate(self.pyr_shapes):#bilinear, bicubic
      #print(content_image.shape)
      warp_targets.append(F.interpolate(wimg, size=pyr_shape, mode='bilinear', align_corners=False).detach())
      mask_targets.append(F.interpolate(mimg, size=pyr_shape, mode='bilinear', align_corners=False).detach())
      #self.postp(content_image.data[0].cpu()).save("con%d.png" % sid)
      content_targets.append([A.detach() for A in self.vgg(F.interpolate(content_image, size=pyr_shape, mode='bilinear', align_corners=False), self.content_layers)])
    
    #targets = style_targets + content_targets
    #del content_image
    
    for it_idx, max_iter in enumerate(self.max_iters):
      targets = self.style_targets[it_idx] + content_targets[it_idx]
      opt_img = F.interpolate(opt_img, size=self.pyr_shapes[it_idx], mode='bilinear', align_corners=False)
      opt_img = Variable(opt_img.data.clone(), requires_grad=True)
      
      warp_img = warp_targets[it_idx]
      mask_img = mask_targets[it_idx]
      
      #self.postp(opt_img.data[0].cpu()).save("test1_%d.png" % it_idx)
      #self.postp(warp_img.data[0].cpu()).save("warp1_%d.png" % it_idx)
      #save_image(mask_img, ncol=1, filename=("mask1_%d.png" % it_idx))
      
      #run style transfer
      #max_iter = 150#250
      #show_iter = 10#max_iter#50
      optimizer = optim.LBFGS([opt_img])
      n_iter=[0]
      
      while n_iter[0] <= max_iter:
        def closure():
          optimizer.zero_grad()
          out = self.vgg(opt_img, self.loss_layers)
          layer_losses = [self.weights[a]*self.loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
          layer_losses += [weight_tcl*((mask_img*(opt_img - warp_img))**2).mean()]

          loss = sum(layer_losses)
          loss.backward()
          n_iter[0]+=1
          #print loss
          #if n_iter[0] % show_iter == (show_iter - 1):
          #  print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
          #  ll = [l.item() for l in layer_losses]
          #  print(ll)
          #  print("r21: %04d, r31: %04d, r41: %04d, r42: %04d, tcl: %04d" % [l.item() for l in layer_losses])
          return loss
          
        optimizer.step(closure)
        
      #self.postp(opt_img.data[0].cpu()).save("test%d.png" % it_idx)
      #self.postp(warp_img.data[0].cpu()).save("warp%d.png" % it_idx)
      #save_image(mask_img, ncol=1, filename=("mask%d.png" % it_idx))
    
    return opt_img.detach()


def eval_sintel(net, args):
  sintel_dir="G:/Datasets/MPI-Sintel-complete/"
  #sintel_dir="/srv/local/tomstrident/datasets/MPI-Sintel-complete/"
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  out_path = os.getcwd() + "/eval_sintel/" + str(args.weight_tcl) + "/"
  raft_model = initRaftModel(args)
  
  pyr_shapes = [(109, 256), (218, 512), (436, 1024)]
  net.set_shapes(pyr_shapes)
  
  num_domains = 4
  net.batch_size = 1
  
  #transform = []
  #transform.append(transforms.ToTensor())
  #transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
  #transform = transforms.Compose(transform)
  transform = T.Compose([#T.Resize((436, 1024)),
                         T.ToTensor(),
                         T.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                         T.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
                         T.Lambda(lambda x: x.mul_(255))])
  
  transform2 = T.Compose([#T.Resize((436, 1024)),
                         #T.ToTensor(),
                         T.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                         T.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
                         T.Lambda(lambda x: x.mul_(255))])
  
  train_dir = os.path.join(sintel_dir, "training", "final")
  train_list = os.listdir(train_dir)
  train_list.sort()
  
  test_dir = os.path.join(sintel_dir, "test", "final")
  test_list = os.listdir(test_dir)
  test_list.sort()
  
  video_list = [os.path.join(train_dir, vid) for vid in train_list]
  video_list += [os.path.join(test_dir, vid) for vid in test_list]
  
  #video_list = video_list[:1]
  
  vid_list = train_list + test_list
  tcl_st_dict = {}
  tcl_lt_dict = {}
  
  tcl_st_dict = OrderedDict()
  tcl_lt_dict = OrderedDict()
  dt_dict = OrderedDict()
  
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
      
      net.set_style(y - 1)
      x_fake = []
      styled_past = []
      #imgs_past = []
      
      for i, imgs in enumerate(tqdm(loader, total=len(loader))):
        img, img_last, img_past = imgs
        
        img = img.to(device)
        img_last = img_last.to(device)
        img_past  = img_past.to(device)
        
        
        if i > 0:
          ff_last = computeRAFT(raft_model, img_last, img)
          bf_last = computeRAFT(raft_model, img, img_last)
          mask_last = fbcCheckTorch(ff_last, bf_last)
          #pre = warp(styled_past[-1], bf_last)
          #pre = mask_last*warp(styled_past[-1], bf_last)
          pre = mask_last*warp(styled_past[-1], bf_last) + (1 - mask_last)*img
          #pre = img
          #net.postp(pre.data[0].cpu()).save("test%d.png" % i)
        else:
          pre = img
          mask_last = torch.zeros((1,) + img.shape[2:]).to(device).unsqueeze(1)
        
        #pre = transform2(torch.randn(img.size())[0]).unsqueeze(0).to(device)
        
        #pre = img
        mask_last = torch.zeros((1,) + img.shape[2:]).to(device).unsqueeze(1)
        
        '''
        if i > 1:
          ff_last = computeRAFT(raft_model, imgs_past[-2], img)
          bf_last = computeRAFT(raft_model, img, imgs_past[-2])
          mask_last = torch.clamp(mask_last - fbcCheckTorch(ff_last, bf_last), 0.0, 1.0)
          pre = mask_last*warp(styled_past[-2], bf_last) + (1 - mask_last)*pre
          #net.postp(pre.data[0].cpu()).save("test%d.png" % i)
          #blah
        '''
  
        #save_image(img[0], ncol=1, filename="blah.png")
        
        t_start = time.time()
        x_fake = net.run(pre, img, y - 1, mask_last, args.weight_tcl)
        t_end = time.time()
        
        #save_image(x_fake[0], ncol=1, filename="blah2.png")
        #blah
        
        dt_vals.append((t_end - t_start)*1000)
        styled_past.append(x_fake)
        #imgs_past.append(img)
          
        if i > 0:
          tcl_st = ((mask_last*(x_fake - pre))**2).mean()**0.5
          tcl_st_vals.append(tcl_st.cpu().numpy())
          #blah
          #tcl_st = computeTCL(net, raft_model, x_fake, img, img_last, y - 1)
          #tcl_st_vals.append(tcl_st.cpu().numpy())
        
        if i >= 5:
          ff_past = computeRAFT(raft_model, img_past, img)
          bf_past = computeRAFT(raft_model, img, img_past)
          mask_past = fbcCheckTorch(ff_past, bf_past)
          warp_past = warp(styled_past[0], bf_past)
          tcl_lt = ((mask_past*(x_fake - warp_past))**2).mean()**0.5
          tcl_lt_vals.append(tcl_lt.cpu().numpy())
          styled_past.pop(0)
          #imgs_past.pop(0)
        
        filename = os.path.join(vid_path, "frame_%04d.png" % i)
        #save_image(x_fake[0], ncol=1, filename=filename)
        if y - 1 == 2:
          out_img = net.postp2(x_fake.data[0].cpu())
        else:
          out_img = net.postp(x_fake.data[0].cpu())
        out_img.save(filename)
        
      tcl_st_dict["TCL-ST_" + key] = float(np.array(tcl_st_vals).mean())
      tcl_lt_dict["TCL-LT_" + key] = float(np.array(tcl_lt_vals).mean())
      dt_dict["DT_" + key] = float(np.array(dt_vals).mean())
  
  save_dict_as_json("TCL-ST", tcl_st_dict, out_path, num_domains)
  save_dict_as_json("TCL-LT", tcl_lt_dict, out_path, num_domains)
  save_dict_as_json("DT", dt_dict, out_path, num_domains)

#==============================================================================

def eval_fc2(net, args):
  print('Calculating evaluation metrics...')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  data_dir = "G:/Datasets/FC2/DATAFiles/"
  style_dir = "G:/Datasets/FC2/styled-files/"
  temp_dir = "G:/Datasets/FC2/styled-files3/"
  
  #data_dir = "/srv/local/tomstrident/datasets/FC2/DATAFiles/"
  #style_dir = "/srv/local/tomstrident/datasets/FC2/styled-files/"
  #temp_dir = "/srv/local/tomstrident/datasets/FC2/styled-files3/"
  
  eval_dir = os.getcwd() + "/eval_fc2/" + str(args.weight_tcl) + "/"
  
  num_workers = 0
  net.batch_size = 1#args.batch_size
  
  pyr_shapes = [(64, 64), (128, 128), (256, 256)]
  net.set_shapes(pyr_shapes)
  
  transform = T.Compose([#T.Resize(pyr_shapes[-1]),
                         T.ToTensor(),
                         T.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                         T.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
                         T.Lambda(lambda x: x.mul_(255))])

  domains = os.listdir(style_dir)
  domains.sort()
  num_domains = len(domains)
  print('Number of domains: %d' % num_domains)
  print("Batch Size:", args.batch_size)
  
  _, eval_loader = get_loaderFC2(data_dir, style_dir, temp_dir, transform, args.batch_size, num_workers, num_domains)
  
  generate_new = True
  
  tcl_dict = {}
  # prepare
  for d in range(1, num_domains):
    src_domain = "style0"
    trg_domain = "style" + str(d)
    
    t1 = '%s2%s' % (src_domain, trg_domain)
    t2 = '%s2%s' % (trg_domain, src_domain)
    
    tcl_dict[t1] = []
    tcl_dict[t2] = []
    
    if generate_new:
      create_task_folders(eval_dir, t1)
      #create_task_folders(eval_dir, t2)

  # generate
  for i, x_src_all in enumerate(tqdm(eval_loader, total=len(eval_loader))):
    x_real, x_real2, y_org, x_ref, y_trg, mask, flow = x_src_all
    
    x_real = x_real.to(device)
    x_real2 = x_real2.to(device)
    y_org = y_org.to(device)
    x_ref = x_ref.to(device)
    y_trg = y_trg.to(device)
    mask = mask.to(device)
    flow = flow.to(device)
    
    mask_zero = torch.zeros(mask.shape).to(device)
    
    N = x_real.size(0)
    #y = y_trg.cpu().numpy()

    for k in range(N):
      y_org_np = y_org[k].cpu().numpy()
      y_trg_np = y_trg[k].cpu().numpy()
      src_domain = "style" + str(y_org_np)
      trg_domain = "style" + str(y_trg_np)
      
      if src_domain == trg_domain or y_trg_np == 0:
        continue
      
      task = '%s2%s' % (src_domain, trg_domain)
      net.set_style(y_trg_np - 1)
      
      x_fake = net.run(x_real, x_real, y_trg_np - 1, mask_zero, args.weight_tcl)
      x_warp = warp(x_fake, flow)
      #x_fake2 = net.run(mask*x_warp  + (1 - mask)*x_real2, x_real2, y_trg_np - 1, mask)
      x_fake2 = net.run(x_warp, x_real2, y_trg_np - 1, mask, args.weight_tcl)
      
      tcl_err = ((mask*(x_fake2 - x_warp))**2).mean(dim=(1, 2, 3))**0.5
      
      tcl_dict[task].append(tcl_err[k].cpu().numpy())

      path_ref = os.path.join(eval_dir, task + "/ref")
      path_fake = os.path.join(eval_dir, task + "/fake")
      
      if generate_new:
        filename = os.path.join(path_ref, '%.4i.png' % (i*args.batch_size+(k+1)))
        if y_trg_np - 1 == 2:
          out_img = net.postp2(x_ref.data[0].cpu())
        else:
          out_img = net.postp(x_ref.data[0].cpu())
        out_img.save(filename)
      
      filename = os.path.join(path_fake, '%.4i.png' % (i*args.batch_size+(k+1)))
      if y_trg_np - 1 == 2:
        out_img = net.postp2(x_fake.data[0].cpu())
      else:
         out_img = net.postp(x_fake.data[0].cpu())
      out_img.save(filename)

  # evaluate
  print("computing fid, lpips and tcl")

  tasks = [dir for dir in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, dir))]
  tasks.sort()

  # fid and lpips
  fid_values = OrderedDict()
  #lpips_dict = OrderedDict()
  tcl_values = OrderedDict()
  for task in tasks:
    print(task)
    path_ref = os.path.join(eval_dir, task + "/ref")
    path_fake = os.path.join(eval_dir, task + "/fake")

    tcl_data = tcl_dict[task]
      
    print("TCL", len(tcl_data))
    tcl_mean = np.array(tcl_data).mean()
    print(tcl_mean)
    tcl_values['TCL_%s' % (task)] = float(tcl_mean)

    print("FID")
    fid_value = calculate_fid_given_paths(paths=[path_ref, path_fake], img_size=256, batch_size=args.batch_size)
    fid_values['FID_%s' % (task)] = fid_value
  
  # calculate the average FID for all tasks
  fid_mean = 0
  for key, value in fid_values.items():
    fid_mean += value / len(fid_values)

  fid_values['FID_mean'] = fid_mean

  # report FID values
  filename = os.path.join(eval_dir, 'FID.json')
  utils.save_json(fid_values, filename)
  
  # calculate the average TCL for all tasks
  tcl_mean = 0
  for _, value in tcl_values.items():
    tcl_mean += value / len(tcl_values)
  
  tcl_values['TCL_mean'] = float(tcl_mean)

  # report TCL values
  filename = os.path.join(eval_dir, 'TCL.json')
  utils.save_json(tcl_values, filename)
    
def main(args):
  obst = OBST()
  
  if args.mode == 'sintel':
    eval_sintel(obst, args)
  
  if args.mode == 'fc2':
    eval_fc2(obst, args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--mode', type=str, default='sintel', choices=['sintel', 'fc2'], help='Eval mode')
  parser.add_argument('--weight_tcl', type=int, default=0,  help='Batch size for fc2_eval')#2000
  parser.add_argument('--batch_size', type=int, default=1,  help='Batch size for fc2_eval')

  #raft
  parser.add_argument('--model', default='models/raft-chairs.pth', help="restore checkpoint")
  parser.add_argument('--path', default='demo-frames', help="dataset for evaluation")
  parser.add_argument('--small', action='store_true', help='use small model')
  parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
  parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
  
  args = parser.parse_args()
  
  main(args)