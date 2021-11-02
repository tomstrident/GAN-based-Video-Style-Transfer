# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:47:25 2021

@author: Tom
"""

import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
#from core.data_loader import get_eval_loader
from sg2_core import utils

from PIL import Image
import torch.nn.functional as F
from models import create_model

from sg2_core.data_loader import get_loaderFC2

def chunks(lst, n):
  #"""Yield successive n-sized chunks from lst."""
  #for i in range(0, len(lst), n):
  #  yield lst[i:i + n]
  return [lst[i:i + n] for i in range(0, len(lst), n)]

def load_image(imfile, device="cuda"):
  img = np.array(Image.open(imfile)).astype(np.uint8)
  img = torch.from_numpy(img).permute(2, 0, 1).float()
  return img[None].to(device)

def load_images(imgs_path):
  imgs_path_list = os.listdir(imgs_path)
  imgs_path_list.sort()
  return [load_image(os.path.join(imgs_path, img_path)) for img_path in imgs_path_list]

def warp(x, f):
  B, C, H, W = x.size()
        
  xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
  yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
  xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
  yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
  grid = torch.cat((xx ,yy) ,1).float().cuda()
  
  vgrid = torch.autograd.Variable(grid) + f
  vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1) - 1.0
  vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1) - 1.0
  vgrid = vgrid.permute(0 ,2 ,3 ,1)
  
  return F.grid_sample(x, vgrid, align_corners=False)

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

@torch.no_grad()
def evaluate_fc2(args):
  print('Calculating evaluation metrics...')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  data_dir = "G:/Datasets/FC2/DATAFiles/"
  style_dir = "G:/Datasets/FC2/styled-files/"
  temp_dir = "G:/Datasets/FC2/styled-files3/"
  
  eval_dir = os.getcwd() + "/eval_fc2/"
  
  num_workers = 0
  args.batch_size = 4

  domains = os.listdir(style_dir)
  domains.sort()
  num_domains = len(domains)
  print('Number of domains: %d' % num_domains)
  print("Batch Size:", args.batch_size)
  
  _, eval_loader = get_loaderFC2(data_dir, style_dir, temp_dir, args.batch_size, num_workers, num_domains)
  
  model_list = os.listdir(args.checkpoints_dir)
  model_list.sort()
  
  #model = create_model(args)
  #model.setup(args)
  
  #generate_new = True

  #num_files = sum([len(files) for r, d, files in os.walk(args.eval_dir)])
  #print("num_files", num_files, len(eval_loader), (1 + args.num_outs_per_domain)*len(eval_loader)*args.batch_size)

  #if num_files != (1 + args.num_outs_per_domain)*len(eval_loader):
  #shutil.rmtree(args.eval_dir, ignore_errors=True)
  #os.makedirs(args.eval_dir)
  generate_new = True
  
  tcl_dict = {}
  models = []
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
    
    args.name = model_list[d-1]
    model = create_model(args)
    model.setup(args)
    models.append(model)

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
    
    N = x_real.size(0)

    for k in range(N):
      y_org_np = y_org[k].cpu().numpy()
      y_trg_np = y_trg[k].cpu().numpy()
      src_domain = "style" + str(y_org_np)
      trg_domain = "style" + str(y_trg_np)
      
      if src_domain == trg_domain or y_trg_np == 0:
        continue
      
      task = '%s2%s' % (src_domain, trg_domain)

      if y_trg_np != 0:
        y = y_trg_np
        
        x_fake = models[y-1].forward_eval(x_real)
        x_fake2 = models[y-1].forward_eval(x_real2)
      else:
        y = y_org_np
        
        x_fake = models[y-1].forward_eval(x_real, AtoB=False)
        x_fake2 = models[y-1].forward_eval(x_real2, AtoB=False)
      
      x_warp = warp(x_fake, flow)
      tcl_err = ((mask*(x_fake2 - x_warp))**2).mean(dim=(1, 2, 3))**0.5
      
      tcl_dict[task].append(tcl_err[k].cpu().numpy())

      path_ref = os.path.join(eval_dir, task + "/ref")
      path_fake = os.path.join(eval_dir, task + "/fake")

      #if not os.path.exists(path_ref):
      #  os.makedirs(path_ref)
        
      #if not os.path.exists(path_fake):
      #  os.makedirs(path_fake)
      
      if generate_new:
        filename = os.path.join(path_ref, '%.4i.png' % (i*args.batch_size+(k+1)))
        utils.save_image(x_ref[k], ncol=1, filename=filename)
      
      filename = os.path.join(path_fake, '%.4i.png' % (i*args.batch_size+(k+1)))
      utils.save_image(x_fake[k], ncol=1, filename=filename)

      #filename = os.path.join(args.eval_dir, task + "/tcl_losses.txt")
      #with open(filename, "a") as text_file:
      #  text_file.write(str(tcl_err[k].cpu().numpy()) + "\n")

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
    #path_tcl = os.path.join(eval_dir, task + "/tcl_losses.txt")
  
    #fake_group = load_images(path_fake)
    
    #with open(path_tcl, "r") as text_file:
    #  tcl_data = text_file.read()
    
    #tcl_data = tcl_data.split("\n")[:-1]
    #tcl_data = [float(td) for td in tcl_data]
    tcl_data = tcl_dict[task]
      
    print("TCL", len(tcl_data))
    tcl_mean = np.array(tcl_data).mean()
    print(tcl_mean)
    tcl_values['TCL_%s' % (task)] = float(tcl_mean)
    
    '''
    lpips_values = []
    fake_chunks = chunks(fake_group, 1)
    for cidx in range(len(fake_chunks)):
      lpips_value = calculate_lpips_given_images(fake_chunks[cidx])
      lpips_values.append(lpips_value)
    
    
    print("LPIPS")
    # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
    lpips_mean = np.array(lpips_values).mean()
    lpips_dict['LPIPS_%s' % (task)] = lpips_mean
    '''

    print("FID")
    fid_value = calculate_fid_given_paths(paths=[path_ref, path_fake], img_size=256, batch_size=args.batch_size)
    fid_values['FID_%s' % (task)] = fid_value
  
  '''
  # calculate the average LPIPS for all tasks
  lpips_mean = 0
  for _, value in lpips_dict.items():
      lpips_mean += value / len(lpips_dict)
  lpips_dict['LPIPS_mean'] = lpips_mean

  # report LPIPS values
  filename = os.path.join(args.eval_dir, 'LPIPS.json')
  utils.save_json(lpips_dict, filename)'''
  
  # calculate the average FID for all tasks
  fid_mean = 0
  #fid_means = [[], [], []]
  for key, value in fid_values.items():
    #for d in range(1, num_domains):
    #  if str(d) in key:
    #    fid_means[d-1].append(value)
    fid_mean += value / len(fid_values)
  
  #for d in range(1, num_domains):
  #  fid_values['FID_s%d_mean' % d] = np.array(fid_means[d-1]).mean()
  
  fid_values['FID_mean'] = fid_mean

  # report FID values
  filename = os.path.join(eval_dir, 'FID.json')
  utils.save_json(fid_values, filename)
  
  # calculate the average TCL for all tasks
  tcl_mean = 0
  #tcl_means = [[], [], []]
  for _, value in tcl_values.items():
    #for d in range(1, num_domains):
    #  if str(d) in key:
    #    tcl_means[d-1].append(value)
    #print(value, len(tcl_values))
    tcl_mean += value / len(tcl_values)
  #print(tcl_mean)
  #for d in range(1, num_domains):
  #  tcl_values['TCL_s%d_mean' % d] = np.array(tcl_means[d-1]).mean()
  
  tcl_values['TCL_mean'] = float(tcl_mean)

  # report TCL values
  filename = os.path.join(eval_dir, 'TCL.json')
  utils.save_json(tcl_values, filename)




  
  