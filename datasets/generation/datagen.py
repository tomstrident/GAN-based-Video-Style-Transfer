# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:57:20 2020

@author: Tom
"""

import time
import os 

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict

from dataload import getGatysLoader, getFC2Loader

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

#pyr_shapes = [512, 256, 128, 64]
#max_iters = [25, 50, 75]

#pyr_shapes = [512]
#max_iters = [200]

pyr_shapes = [128, 256, 512]
max_iters = [50, 40, 30]

'''
# pre and post processing for images
prep = transforms.Compose([transforms.Scale(512),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
'''
preps = []
for pyr_shape in pyr_shapes:
  preps.append(transforms.Compose([transforms.Scale(pyr_shape),
               transforms.ToTensor(),
               transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),#subtract imagenet mean                              
               transforms.Lambda(lambda x: x.mul_(255))]))
  

postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb2 = transforms.Compose([transforms.ToPILImage(),
                             transforms.Scale(256),
                             transforms.Grayscale(num_output_channels=3)])
postpb = transforms.Compose([transforms.ToPILImage(),
                             transforms.Scale(256)])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img
    
def postp2(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb2(t)
    return img

batch_size = 32
#dset_img_dir = "F:\\Datasets\\GAN\\train\\data\\"
#output_dir = "F:\\Datasets\\GAN\\train\\style-set2\\"
#dset_img_dir = "F:\\Datasets\\FC2\\DATAFiles\\"
#output_dir = "F:\\Datasets\\FC2\\styled-files\\"
dset_img_dir = "/home/tomstrident/datasets/FC2/DATAFiles/"
output_dir =   "/home/tomstrident/datasets/FC2/styled-files3/"
style_img_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'


dset_img_list = os.listdir(dset_img_dir)
dset_img_list.sort()

style_img_list = os.listdir(style_img_dir)
style_img_list.sort()

#dataloader
#dset_loader = getGatysLoader(dset_img_dir, batch_size)
dset_loader = getFC2Loader(dset_img_dir, batch_size, num_w=8)

#get network
vgg = VGG()
model_weights = torch.load(model_dir + 'vgg_conv.pth')
del model_weights["conv5_2.weight"]
del model_weights["conv5_2.bias"]
del model_weights["conv5_3.weight"]
del model_weights["conv5_3.bias"]
del model_weights["conv5_4.weight"]
del model_weights["conv5_4.bias"]

vgg.load_state_dict(model_weights)
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

#load images, ordered as [style_image, content_image]
#img_dirs = [image_dir, image_dir]
#img_names = ['vangogh_starry_night.jpg', 'Tuebingen_Neckarfront.jpg']
#imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
#imgs_torch = [prep(img) for img in imgs]
#if torch.cuda.is_available():
#    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
#else:
#    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
#style_image, content_image = imgs_torch

#define layers, loss functions, weights and compute optimization targets
#style_layers = ['r11','r21','r31','r41', 'r51']
style_layers = ['r21','r31','r41']
content_layers = ['r42']
loss_layers = style_layers + content_layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
    
#these are good weights settings:
#style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
beta = 1e2 # 1e2
style_weights = [beta/n**2 for n in [128,256,512]]
content_weights = [1e0]
weights = style_weights + content_weights

style_len = len(style_img_list)
dset_len = len(dset_loader)

out_dir_c = output_dir + "style0/"

if not os.path.exists(out_dir_c):
  os.mkdir(out_dir_c)

for sid, style_img in enumerate(style_img_list):
  #if sid < 2:
  #  continue
  out_dir = output_dir + "style" + str(sid+1) + "/"
  #print(out_dir)
  #print(style_img)
  #blah
  
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  
  style_targets = []
  for prep in preps:
    style_image = prep(Image.open(style_img_dir + style_img))
    style_image = [style_image.unsqueeze(0) for b in range(batch_size)]
    style_image = torch.cat(style_image, dim=0)
    style_image = Variable(style_image.cuda())
    #style_image = Variable(style_image.unsqueeze(0).cuda())
  
    #compute optimization targets
    style_targets.append([GramMatrix()(A).detach() for A in vgg(style_image, style_layers)])
    del style_image

  dset_iter = iter(dset_loader)

  #for dset_img in dset_img_list:
  for i in range(len(dset_loader)):
    t_start = time.time()
    
    try:
      content_image, image_id = next(dset_iter)
      if content_image.shape[0] != batch_size:
        raise Exception()
    except:
      print("data_iter end")
      break

    if os.path.exists(out_dir + image_id[0]):
      continue
    #if i < 1200:
    #  continue

    content_image = content_image.to("cuda")
    
    # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
    opt_img = Variable(content_image.data.clone(), requires_grad=True)
    
    #content_image = prep(Image.open(dset_img_dir + dset_img))
    #content_image = Variable(content_image.unsqueeze(0).cuda())
    #print(content_image.shape)
    
    content_targets = []
    for pyr_idx, pyr_shape in enumerate(pyr_shapes):#bilinear, bicubic
      #print(content_image.shape)
      content_targets.append([A.detach() for A in vgg(F.interpolate(content_image, size=(pyr_shape, pyr_shape), mode='bilinear', align_corners=False), content_layers)])
    
    #targets = style_targets + content_targets
    #del content_image
    
    for it_idx, max_iter in enumerate(max_iters):
      targets = style_targets[it_idx] + content_targets[it_idx]
      opt_img = F.interpolate(opt_img, size=(pyr_shapes[it_idx], pyr_shapes[it_idx]), mode='bilinear', align_corners=False)
      opt_img = Variable(opt_img.data.clone(), requires_grad=True)
      #run style transfer
      #max_iter = 150#250
      show_iter = max_iter#50
      optimizer = optim.LBFGS([opt_img])
      n_iter=[0]
      
      while n_iter[0] <= max_iter:
        def closure():
          optimizer.zero_grad()
          out = vgg(opt_img, loss_layers)
          layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
          loss = sum(layer_losses)
          loss.backward()
          n_iter[0]+=1
          #print loss
          if n_iter[0] % show_iter == (show_iter - 1):
            print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
          return loss
          
        optimizer.step(closure)
    
    for j, iid in enumerate(image_id):
      #save content as style0
      if sid == 0:
        out_img = postp(content_image.data[j].cpu())#.squeeze()
        out_img.save(out_dir_c + iid)
      
      #save styles
      if sid == 2:
        out_img = postp2(opt_img.data[j].cpu())#.squeeze()
      else:
        out_img = postp(opt_img.data[j].cpu())#.squeeze()
      #out_img = postp(content_image.data[j].cpu())#.squeeze()
      out_img.save(out_dir + iid)
    
    t_end = time.time()
    print("Style: %d/%d, Iter: %d/%d, elapsed time: %f" % (sid + 1, style_len, i + 1, dset_len, t_end - t_start))

