# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:05:59 2021

@author: Tom
"""

import torch
import numpy as np
import torch.nn.functional as F

def gradient(x):
  dx = (F.pad(x, (0, 1, 0, 0))[:,:,1:] - F.pad(x, (1, 0, 0, 0))[:,:,:-1])/2
  dy = (F.pad(x, (0, 0, 0, 1))[:,1:,:] - F.pad(x, (0, 0, 1, 0))[:,:-1,:])/2

  return torch.stack([dx, dy]) 

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

def fbcCheckTorch(ff, bf, device="cuda"):
  #wf = warp(ff, bf)
  B, C, H, W = bf.size()
  
  mask = torch.ones((B, H, W)).to(device)
  z = torch.Tensor([0.0]).to(device)
  
  #norm_wb = torch.norm(wf + bf, dim=1)**2
  #norm_w = torch.norm(wf, dim=1)**2
  norm_b = torch.norm(bf, dim=1)**2
  
  #occ = norm_wb > 0.01*(norm_w + norm_b) + 0.5

  grad_u = gradient(bf[:,0,:,:])
  grad_v = gradient(bf[:,1,:,:])

  norm_u = torch.norm(grad_u, dim=0)**2.0
  norm_v = torch.norm(grad_v, dim=0)**2.0
  
  mob = norm_u + norm_v > 0.01*norm_b + 0.002
  
  #mask = torch.where(occ, z, mask)
  mask = torch.where(mob, z, mask)
  
  return mask.unsqueeze(1)