# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.parallel

from torch.utils.data import DataLoader
from network import Vgg16, Vgg19
from datasets import FlyingChairs2Dataset, Hollywood2Dataset, COCODataset, SintelDataset

from network import FastStyleNet

import cv2
import time
import imageio
import numpy as np
from skimage import io, transform

from PIL import Image

from collections import OrderedDict
from tqdm import tqdm

import random
import sys
sys.path.append('raft')
from raft.raft import RAFT
from flowtools import fbcCheckTorch, warp

from torch.utils import data
from torchvision import transforms
from utils.utils import InputPadder
import torchvision.utils as vutils

import json
import shutil
from sg2_core.data_loader import get_loaderFC2
from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
#from core.data_loader import get_eval_loader
from sg2_core import utils

def denormalize(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)

def save_image(x, ncol, filename, gray=False):
  #x = denormalize(x)
  if gray:
    gtr = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()])
    x = gtr(x)
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

def computeTCL(net, model, s_trg, img_fake, img1, img2):
  ff_last = computeRAFT(model, img2, img1)
  bf_last = computeRAFT(model, img1, img2)
  mask_last = fbcCheckTorch(ff_last, bf_last)
  warp_last = warp(net.generator(img2, s_trg), bf_last)
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

# =============================================================================
class FastStyle():
  def __init__(self, debug=True):
    #self.train_dir = 'F:/runs/'
    #self.train_dir = '/home/tomstrident/projects/LBST/runs/'
    self.train_dir = 'G:/Code/LBST/runs/'
    self.debug = debug
    self.device = 'cuda'
    self.method = []
    
    self.VGG16_MEAN = [0.485, 0.456, 0.406]
    self.VGG16_STD = [0.229, 0.224, 0.225]
    
    #self.sid_styles = ['autoportrait', 'edtaonisl', 'composition', 'edtaonisl', 'udnie', 'starry_night']#'candy', 
    self.sid_styles = ['s1_starry_night', 's2_the_scream', 's3_take_on_me']#'candy', 
    
    style_grid = np.arange(0, len(self.sid_styles), dtype=np.float32)
    self.style_id_grid = torch.Tensor(style_grid).to(self.device).float()
    
    if debug and not os.path.exists("debug/"):
      os.mkdir("debug/")
  
  def vectorize_parameters(self, params, n_styles):
    vec_pararms = [p*np.ones(n_styles) for p in np.array(params)]
    return np.array(vec_pararms).T
  
  def concat_id(self, params):
    run_id = ""
    
    for j, p in enumerate(params):
      for pi in p:
        run_id += "_" + self.loss_letters[j] + ("%d" % np.log10(pi))

    return run_id + "/"
  
  def train(self, sid=2, epochs=3, emphasis_parameter=[1e0, 1e1], 
            batchsize=16, learning_rate=1e-3, 
            dset='FC2'):
    
    if isinstance(sid, list):
      styles = [self.sid_styles[sidx] for sidx in sid]
      run_id = "msid%d_ep%d_bs%d_lr%d" % (len(sid), epochs, batchsize, np.log10(learning_rate))
      emphasis_parameter = self.vectorize_parameters(emphasis_parameter, len(sid))
    else:
      styles = [self.sid_styles[sid]]
      run_id = "sid%d_ep%d_bs%d_lr%d" % (sid, epochs, batchsize, np.log10(learning_rate))
      emphasis_parameter = self.vectorize_parameters(emphasis_parameter, 1)
    
    #self.train_dir = self.train_dir[:8] + dset + '/' + self.method + '/'
    self.train_dir = self.train_dir + dset + '/' + self.method + '/'
    run_id = self.setup_method(run_id, emphasis_parameter.T)
    adv_train_dir = self.train_dir + run_id
    print(adv_train_dir)
    
    if not os.path.exists(adv_train_dir):
      os.makedirs(adv_train_dir)
     
    if os.path.exists(adv_train_dir + '/epoch_' + str(epochs-1) + '.pth'):
      print('Warning: config already exists! Returning ...')
      return
    
    self.prep_training(batch_sz=batchsize, styles=styles, dset=dset)
    self.adam = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    loss_list = []
    
    n_styles = len(self.styles)
    style_grid = np.arange(0, n_styles)
    style_id_grid = torch.LongTensor(style_grid).to(self.device)
    
    for epoch in range(epochs):
      for itr, (imgs, masks, flows) in enumerate(self.dataloader):
        
        imgs = torch.split(imgs, 3, dim=1)
        
        self.prep_adam(itr)
        
        if n_styles > 1:
          style_id = style_id_grid[np.random.randint(0, n_styles)]
        else:
          style_id = 0
        
        losses, styled_img, loss_string = self.train_method(imgs, masks, flows, emphasis_parameter[style_id], style_id)
        
        self.adam.step()
        
        if (itr+1)%1000 == 0:
          torch.save(self.model.state_dict(), '%sfinal_epoch_%d_itr_%d.pth' % (adv_train_dir, epoch, itr//1000))

        if (itr)%1000 == 0 and self.debug:
          imageio.imsave('debug/%d_%d_img1.png' % (epoch, itr), imgs[0].cpu().numpy()[0].transpose(1,2,0))
          imageio.imsave('debug/%d_%d_styled_img1.png' % (epoch, itr), styled_img.detach().cpu().numpy()[0].transpose(1,2,0))
          
        out_string = "[%d/%d][%d/%d] sid%d" % (epoch, epochs, itr, len(self.dataloader), style_id)
        print(out_string + loss_string)
        loss_list.append(torch.FloatTensor(losses).detach().cpu().numpy())

      torch.save(self.model.state_dict(), '%sepoch_%d.pth' % (adv_train_dir, epoch))

    loss_list = np.array(loss_list)
    np.save(adv_train_dir + "loss_list.npy", loss_list)

  #============================================================================
  def infer(self, sid, n_styles, epochs, n_epochs, emphasis_parameter,
            batchsize=16, learning_rate=1e-3,
            dset='FC2', sintel_id='temple_2', sintel_path='D:/Datasets/', 
            vid_fps=20, out_img_path=None, out_img_num=[10]):
    
    if n_styles > 1:
      run_id = "msid%d_ep%d_bs%d_lr%d" % (n_styles, epochs, batchsize, np.log10(learning_rate))
      
    else:
      run_id = "sid%d_ep%d_bs%d_lr%d" % (sid, epochs, batchsize, np.log10(learning_rate))
      
    emphasis_parameter = self.vectorize_parameters(emphasis_parameter, n_styles)
    
    #self.train_dir = self.train_dir[:8] + dset + '/' + self.method + '/'
    self.train_dir = self.train_dir + dset + '/' + self.method + '/'
    run_id = self.setup_method(run_id, emphasis_parameter.T)
    #infer_id = run_id[:4] + str(sid) + run_id[5:-1]
    
    print(self.train_dir + run_id + 'epoch_' + str(n_epochs) + '.pth')
    self.model.load_state_dict(torch.load(self.train_dir + run_id + 'epoch_' + str(n_epochs) + '.pth'))
    
    writer = imageio.get_writer('styled_' + self.method + str(sid) + '.mp4', fps=vid_fps)
    dataloader = DataLoader(SintelDataset(sintel_path, sintel_id), batch_size=1)
      
    warped = []
    mask = []
    
    cst_list = []
    lt_cst_list = []
    
    ft_count = []
    styled_list = []
    
    #debug_path = 'C:/Users/Tom/Documents/Python Scripts/Masters Project/debug/'

    style_grid = np.arange(0, len(self.sid_styles), dtype=np.float32)
    style_id_grid = torch.Tensor(style_grid).to(self.device).float()
    style_id = style_id_grid[sid]

    for itr, (frame, mask, flow, lt_data) in enumerate(dataloader):

      if itr > 0:
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        warped = self.warp_image(styled_list[-1], flow)
      
      t_start = time.time()
      torch_output = self.infer_method((frame, mask, warped), style_id)
      t_end = time.time()
      
      ft_count.append(t_end - t_start)
      
      torch_output = torch.clamp(torch_output, 0.0, 1.0)
      styled_frame = torch_output[0].permute(1, 2, 0).detach().cpu().numpy()
      
      #imageio.imwrite(debug_path + '/img' + str(itr) + '.png', (styled_frame*255.0).astype(np.uint8))
      
      if itr > 0:
        #imageio.imwrite(debug_path + '/warp' + str(itr) + '.png', (warped*255.0).astype(np.uint8))
        #imageio.imwrite(debug_path + '/mask' + str(itr) + '.png', (mask*255.0).astype(np.uint8))
        mask = mask[0].permute(1, 2, 0).cpu().numpy()
        cst = ((mask*(warped - styled_frame))**2).mean()
        cst_list.append(cst)
        #print('FPS:', 1/ft_count[-1], 'CST:', cst_list[-1])
      
      styled_list.append(styled_frame)
      
      lt_len = 5
      if not (itr - lt_len < 0 or itr == len(dataloader) - 1):
        lt_flow, lt_mask = lt_data
        lt_flow = lt_flow[0].permute(1, 2, 0).cpu().numpy()
        lt_mask = lt_mask[0].permute(1, 2, 0).cpu().numpy()
        f_idx2 = itr-lt_len+1
        #imageio.imwrite(debug_path + '/styled_frame2.png', (styled_list[f_idx1]*255.0).astype(np.uint8))
        #imageio.imwrite(debug_path + '/styled_frame1.png', (styled_list[f_idx2]*255.0).astype(np.uint8))
        warped = self.warp_image(styled_list[f_idx2], lt_flow)
        #imageio.imwrite(debug_path + '/warp' + '.png', (warped*255.0).astype(np.uint8))
        #imageio.imwrite(debug_path + '/wmask' + '.png', (lt_mask*255.0).astype(np.uint8))
        
        lt_cst = ((lt_mask[0]*(warped - styled_frame))**2).mean()
        lt_cst_list.append(lt_cst)
      
      real_fid = len(dataloader) - 1 - itr
      if out_img_path != None and real_fid in out_img_num:
        #imageio.imwrite(self.train_dir + infer_path + '_c.png', (np_f*255.0).astype(np.uint8))
        print(out_img_path + dset + "_" + run_id[:-1] + "_" + str(real_fid) + ".png")
        imageio.imwrite(out_img_path + dset + "_" + run_id[:-1] + "_" + str(real_fid) + ".png", (styled_frame*255.0).astype(np.uint8))
      
      cv2.imshow('frame', styled_frame[:,:,[2, 1, 0]])
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      #writer.append_data((styled_frame*255.0).astype(np.uint8))
    
    cv2.destroyAllWindows()
    
    for styled_frame in styled_list[::-1]:
      writer.append_data((styled_frame*255.0).astype(np.uint8))
    
    writer.close()
    
    ft_count = np.array(ft_count[3:])
    fps_count = np.array([1/x for x in ft_count])
    
    avg_ft = ft_count.mean()
    avg_fps = fps_count.mean()
    
    #avg_ft = ft_count.mean()
    #opl_ft = np.percentile(np.sort(ft_count), 1)
    
    #avg_fps = fps_count.mean()
    #opl_fps = np.percentile(np.sort(fps_count), 1)
    
    #oph_ft = self.high_percentile(ft_count, 5)
    #opl_fps2 = self.high_percentile(fps_count, 5)
    
    mse_cst = (np.array(cst_list).mean())**0.5
    mse_lt_cst = (np.array(lt_cst_list).mean())**0.5
    
    print('consistency mse:', mse_cst)
    print('lt consistency mse:', mse_lt_cst)
    print('avg ft:', avg_ft*1000, avg_fps)
    #print('opl ft:', opl_ft*1000, 1/opl_ft, oph_ft, opl_fps, opl_fps2)
    
    return avg_ft*1000, avg_fps, mse_cst, mse_lt_cst
  
  '''
  infer(self, sid, n_styles, epochs, n_epochs, emphasis_parameter,
            batchsize=16, learning_rate=1e-3,
            dset='FC2', sintel_id='temple_2', sintel_path='D:/Datasets/', 
            vid_fps=20, out_img_path=None, out_img_num=[10]):
  '''
  def evaluate_sintel(self, args, n_styles, epochs, n_epochs, emphasis_parameter, 
               sintel_dir="D:/Datasets/MPI-Sintel-complete/",
               batchsize=16, learning_rate=1e-3, dset='FC2'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_path = "G:/Code/LBST/eval_sintel/" + self.method + "/"
    raft_model = initRaftModel(args)

    num_domains = 4
    
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
    
    #video_list = [os.path.join(train_dir, vid) for vid in train_list]
    #video_list += [os.path.join(test_dir, vid) for vid in test_list]
    
    video_list = [os.path.join(train_dir, "alley_2"), 
                  os.path.join(train_dir, "market_6"), 
                  os.path.join(train_dir, "temple_2")]
    
    #vid_list = train_list + test_list
    vid_list = ["alley_2", "market_6", "temple_2"]

    tcl_st_dict = {}
    tcl_lt_dict = {}
    
    tcl_st_dict = OrderedDict()
    tcl_lt_dict = OrderedDict()
    dt_dict = OrderedDict()
    
    #emphasis_parameter = self.vectorize_parameters(emphasis_parameter, n_styles)
    tmp_dir = self.train_dir + dset + '/' + self.method + '/'
    tmp_list = os.listdir(tmp_dir)
    tmp_list.sort()
    #run_id = self.setup_method(run_id, emphasis_parameter.T)
    
    if self.method == "ruder":
      self.model = FastStyleNet(3 + 1 + 3, n_styles).to(self.device)
      self.pre_style_model = FastStyleNet(3, n_styles).to(self.device)
    else:
      self.model = FastStyleNet(3, n_styles).to(self.device)
      
    first = True
    
    for j, vid_dir in enumerate(video_list):
      vid = vid_list[j]
  
      #print(vid_dir)
  
      sintel_dset = SingleSintelVideo(vid_dir, transform)
      loader = data.DataLoader(dataset=sintel_dset, batch_size=1, shuffle=False, num_workers=0)
      
      for y in range(1, num_domains):
        y_trg = torch.Tensor([y])[0].type(torch.LongTensor).to(device)
        key = vid + "_s" + str(y)
        vid_path = os.path.join(out_path, key)
        if not os.path.exists(vid_path):
          os.makedirs(vid_path)
          
        if y == 3:
          gray = True
        else:
          gray = False
      
        tcl_st_vals = []
        tcl_lt_vals = []
        dt_vals = []
        
        #if n_styles > 1:
        #  run_id = "msid%d_ep%d_bs%d_lr%d" % (n_styles, epochs, batchsize, np.log10(learning_rate))
        #else:
        #  run_id = "sid%d_ep%d_bs%d_lr%d" % (y - 1, epochs, batchsize, np.log10(learning_rate))
        
        if n_styles > 1:
          if first:
            self.model.load_state_dict(torch.load(tmp_dir + '/' + tmp_list[y-1] + '/epoch_' + str(n_epochs) + '.pth'))
            first = False
        else:
          #print(tmp_dir + '/' + tmp_list[y-1] + '/epoch_' + str(n_epochs) + '.pth')
          self.model.load_state_dict(torch.load(tmp_dir + '/' + tmp_list[y-1] + '/epoch_' + str(n_epochs) + '.pth'))
        
        if self.method == "ruder":
          pre_style_path = "G:/Code/LBST/runs/johnson/FC2/johnson/sid" + str(y - 1) + "_ep20_bs16_lr-3_a0_b1_d-4/epoch_19.pth"
          self.pre_style_model.load_state_dict(torch.load(pre_style_path))
        
        past_sty_list = []
        
        for i, imgs in enumerate(tqdm(loader, total=len(loader))):
          img, img_last, img_past = imgs
          
          img = img.to(device)
          img_last = img_last.to(device)
          img_past  = img_past.to(device)
          
          #save_image(img[0], ncol=1, filename="blah.png")
          if i > 0:
            ff_last = computeRAFT(raft_model, img_last, img)
            bf_last = computeRAFT(raft_model, img, img_last)
            mask_last = fbcCheckTorch(ff_last, bf_last)
            x_fake_last = past_sty_list[-1]#self.infer_method((img_last, None, None), y_trg - 1)
            warp_last = warp(torch.clamp(x_fake_last, 0.0, 1.0), bf_last)
          else:
            mask_last = None
            warp_last = None
          #mask, x_warp
          
          t_start = time.time()
          x_fake = self.infer_method((img, mask_last, warp_last), y_trg - 1)
          x_fake = torch.clamp(x_fake, 0.0, 1.0)
          t_end = time.time()
          
          past_sty_list.append(x_fake)
          dt_vals.append((t_end - t_start)*1000)
          
          if i > 0:
            tcl_st = ((mask_last*(x_fake - warp_last))**2).mean()**0.5
            tcl_st_vals.append(tcl_st.cpu().numpy())
          
          if i >= 5:
            ff_past = computeRAFT(raft_model, img_past, img)
            bf_past = computeRAFT(raft_model, img, img_past)
            mask_past = fbcCheckTorch(ff_past, bf_past)
            #torch.clamp(self.infer_method((img_past, None, None), y_trg - 1), 0.0, 1.0)
            warp_past = warp(past_sty_list[0], bf_past)
            tcl_lt = ((mask_past*(x_fake - warp_past))**2).mean()**0.5
            tcl_lt_vals.append(tcl_lt.cpu().numpy())
            
            '''
            print(img.shape)
            print(img_past.shape)
            print(warp_past.shape)
            print(x_fake.shape)
            print(past_sty_list[0].shape)
            
            save_image(denormalize(img[0]), ncol=1, filename="blah1.png")
            save_image(denormalize(img_past[0]), ncol=1, filename="blah2.png")
            save_image(warp_past[0], ncol=1, filename="blah3.png")
            save_image(x_fake[0], ncol=1, filename="blah4.png")
            save_image(past_sty_list[0][0], ncol=1, filename="blah5.png")
            save_image(mask_past*warp_past, ncol=1, filename="blah6.png")
            blah'''
            
            past_sty_list.pop(0)
          
          filename = os.path.join(vid_path, "frame_%04d.png" % i)
          save_image(x_fake[0], ncol=1, filename=filename, gray=gray)
          
        tcl_st_dict["TCL-ST_" + key] = float(np.array(tcl_st_vals).mean())
        tcl_lt_dict["TCL-LT_" + key] = float(np.array(tcl_lt_vals).mean())
        dt_dict["DT_" + key] = float(np.array(dt_vals).mean())
    
    save_dict_as_json("TCL-ST", tcl_st_dict, out_path, num_domains)
    save_dict_as_json("TCL-LT", tcl_lt_dict, out_path, num_domains)
    save_dict_as_json("DT", dt_dict, out_path, num_domains)
    
  @torch.no_grad()
  def evaluate_fc2(self, args, n_styles, epochs, n_epochs, emphasis_parameter, 
               batchsize=16, learning_rate=1e-3, dset='FC2'):
    print('Calculating evaluation metrics...')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    data_dir = "G:/Datasets/FC2/DATAFiles/"
    style_dir = "G:/Datasets/FC2/styled-files/"
    temp_dir = "G:/Datasets/FC2/styled-files3/"
    eval_dir = os.getcwd() + "/eval_fc2/" + self.method + "/"
    
    num_workers = 0
    args.batch_size = 4
  
    domains = os.listdir(style_dir)
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)
    print("Batch Size:", args.batch_size)
    
    _, eval_loader = get_loaderFC2(data_dir, style_dir, temp_dir, args.batch_size, num_workers, num_domains)
    
    tmp_dir = self.train_dir + dset + '/' + self.method + '/'
    tmp_list = os.listdir(tmp_dir)
    tmp_list.sort()
    
    models = []
    pre_models = []
    if n_styles > 1:
      model = FastStyleNet(3, n_styles).to(self.device)
      model.load_state_dict(torch.load(tmp_dir + '/' + tmp_list[0] + '/epoch_' + str(n_epochs) + '.pth'))
    else:
      if self.method == "ruder":
        for tmp in tmp_list:
          model = FastStyleNet(3 + 1 + 3, n_styles).to(self.device)
          model.load_state_dict(torch.load(tmp_dir + '/' + tmp + '/epoch_' + str(n_epochs) + '.pth'))
          models.append(model)
          pre_style_path = "G:/Code/LBST/runs/johnson/FC2/johnson/sid" + tmp[3] + "_ep20_bs16_lr-3_a0_b1_d-4/epoch_19.pth"
          model = FastStyleNet(3, n_styles).to(self.device)
          model.load_state_dict(torch.load(pre_style_path))
          pre_models.append(model)
      else:
        for tmp in tmp_list:
          model = FastStyleNet(3, n_styles).to(self.device)
          model.load_state_dict(torch.load(tmp_dir + '/' + tmp + '/epoch_' + str(n_epochs) + '.pth'))
          models.append(model)
          
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
      
      x_real = x_real.to(self.device)
      x_real2 = x_real2.to(self.device)
      y_org = y_org.to(self.device)
      x_ref = x_ref.to(self.device)
      y_trg = y_trg.to(self.device)
      mask = mask.to(self.device)
      flow = flow.to(self.device)
      
      N = x_real.size(0)
  
      for k in range(N):
        y_org_np = y_org[k].cpu().numpy()
        y_trg_np = y_trg[k].cpu().numpy()
        src_domain = "style" + str(y_org_np)
        trg_domain = "style" + str(y_trg_np)
        
        if src_domain == trg_domain or y_trg_np == 0:
          continue
        
        task = '%s2%s' % (src_domain, trg_domain)

        if n_styles > 1:
          self.model = model
        else:
          self.model = models[y_trg_np-1]
          
        if self.method == "ruder":
          self.pre_style_model = pre_models[y_trg_np-1]
        
        x_fake = self.infer_method((x_real, None, None), y_trg[k] - 1)
        #x_fake = torch.clamp(x_fake, 0.0, 1.0)
        x_warp = warp(x_fake, flow)
        x_fake2 = self.infer_method((x_real2, mask, x_warp), y_trg[k] - 1)
        #x_fake2 = torch.clamp(x_fake2, 0.0, 1.0)
        
        tcl_err = ((mask*(x_fake2 - x_warp))**2).mean(dim=(1, 2, 3))**0.5
        
        tcl_dict[task].append(tcl_err[k].cpu().numpy())
  
        path_ref = os.path.join(eval_dir, task + "/ref")
        path_fake = os.path.join(eval_dir, task + "/fake")
        
        if generate_new:
          filename = os.path.join(path_ref, '%.4i.png' % (i*args.batch_size+(k+1)))
          save_image(denormalize(x_ref[k]), ncol=1, filename=filename)
        
        filename = os.path.join(path_fake, '%.4i.png' % (i*args.batch_size+(k+1)))
        save_image(x_fake[k], ncol=1, filename=filename)
  
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
  
  def setup_train(self):
    raise NotImplementedError("Please Implement this method")
  
  def train_method(self):
    raise NotImplementedError("Please Implement this method")
    
  def infer_method(self):
    raise NotImplementedError("Please Implement this method")

  def setup_method(self):
    raise NotImplementedError("Please Implement this method")

  def loadStyles(self, style_name_list, style_size=512):
    styles = []
    
    for i, style_name in enumerate(style_name_list):
      style = io.imread('styles/' + style_name + '.jpg')
      style = torch.from_numpy(transform.resize(style, (style_size, style_size))).to(self.device).permute(2, 0, 1).float().unsqueeze(0)
    
      if self.debug:
        imageio.imsave('debug/0_0_style_' + str(i) + '.png', style.cpu().numpy()[0].transpose(1,2,0))
    
      style = self.normalize(style)
      styled_featuresR = self.vgg(style)
      style_GM = [self.gram_matrix(f) for f in styled_featuresR]
      
      styles.append(style_GM)
    
    return styles

  def load_model(self, model_path):
    print('loading model ...')
    self.model.load_state_dict(torch.load(self.train_dir + model_path))

  def prep_training(self, batch_sz=16, styles=['composition'], dset='FC2'):
    #dset_path = 'F:/Datasets/' + dset + '/'
    dset_path = '/home/tomstrident/datasets/' + dset + '/'
    
    if dset == 'FC2':
      self.dataloader = DataLoader(FlyingChairs2Dataset(dset_path, batch_sz), batch_size=batch_sz)#, num_workers=4
    elif dset == 'HW2':
      self.dataloader = DataLoader(Hollywood2Dataset(dset_path, batch_sz), batch_size=batch_sz)
    elif dset == 'CO2':
      self.dataloader = DataLoader(COCODataset(dset_path, batch_sz), batch_size=batch_sz)
    else:
      assert False, "Invalid dataset specified error!"
    
    self.train_dir = self.train_dir[:5] + dset + '/'
  
    self.L2distance = nn.MSELoss().to(self.device)
    self.L2distancematrix = nn.MSELoss(reduction='none').to(self.device)
    self.vgg = Vgg16().to(self.device)
    #self.vgg = Vgg19().to(self.device)
    
    for param in self.vgg.parameters():
      param.requires_grad = False
    
    self.styles = self.loadStyles(styles)
    self.adam = []

  def prep_adam(self, itr, batch_sz=16):
    self.adam.zero_grad()

    if (itr+1) % np.int32(500 / batch_sz) == 0:
      for param in self.adam.param_groups:
        param['lr'] = max(param['lr']/1.2, 1e-4)
  
  def calc_tv_loss(self, I):
    sij = I[:, :, :-1, :-1]
    si1j = I[:, :, :-1, 1:]
    sij1 = I[:, :, 1:, :-1]
    
    tv_mat1 = torch.norm(sij1 - sij, dim=1)**2
    tv_mat2 = torch.norm(si1j - sij, dim=1)**2
    
    return torch.sum((tv_mat1 + tv_mat2)**0.5)
  
  def load_mp4(self, video_path):
    reader = imageio.get_reader(video_path + '.mp4')
    fps = reader.get_meta_data()['fps']
    num_f = reader.count_frames()
    print(num_f)
    
    return num_f, fps, reader

  def gram_matrix(self, inp):
    b, c, h, w = inp.size()
    features = inp.view(b, c, h*w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(h*w)
  
  def normalize(self, img):
    mean = img.new_tensor(self.VGG16_MEAN).view(-1, 1, 1)
    std = img.new_tensor(self.VGG16_STD).view(-1, 1, 1)
    return (img - mean) / std

  def warp_image(self, A, flow):
    h, w = flow.shape[:2]
    x = (flow[...,0] + np.arange(w)).astype(A.dtype)
    y = (flow[...,1] + np.arange(h)[:,np.newaxis]).astype(A.dtype)
  
    W_m = cv2.remap(A, x, y, cv2.INTER_LINEAR)
  
    return W_m.reshape(A.shape)

  def styleFrame(self, frame, sid):
    style_id = torch.from_numpy(np.float32([sid])).to(self.device).float()[0]
    
    torch_f = torch.from_numpy(frame).to(self.device).permute(2, 0, 1).float().unsqueeze(0)
    torch_m = torch.zeros(1, 1, frame.shape[0], frame.shape[1])
    torch_w = torch_f
    torch_output = self.infer_method((torch_f, torch_m, torch_w), style_id)
    
    torch_output = torch.clamp(torch_output, 0.0, 1.0)
    styled_frame = torch_output[0].permute(1, 2, 0).detach().cpu().numpy()
    
    return styled_frame
  
  def loadModel(self, sid, n_styles, epochs, n_epochs, emphasis_parameter, 
                batchsize=6, learning_rate=1e-3, dset='FC2'):
    
    if n_styles > 1:
      run_id = "msid%d_ep%d_bs%d_lr%d" % (n_styles, epochs, batchsize, np.log10(learning_rate))
      
    else:
      run_id = "sid%d_ep%d_bs%d_lr%d" % (sid, epochs, batchsize, np.log10(learning_rate))
      
    emphasis_parameter = self.vectorize_parameters(emphasis_parameter, n_styles)
    
    self.train_dir = self.train_dir[:8] + dset + '/' + self.method + '/'
    run_id = self.setup_method(run_id, emphasis_parameter.T)
    
    print(self.train_dir + run_id + 'epoch_' + str(n_epochs) + '.pth')
    self.loadModelID(self.train_dir + run_id + 'epoch_' + str(n_epochs) + '.pth')
    
  def loadModelID(self, n_styles, model_id):
    self.model = FastStyleNet(3, n_styles).to(self.device)
    self.model.load_state_dict(torch.load(model_id))