# -*- coding: utf-8 -*-

import torch
from fs_lib import warp
from network import FastStyleNet
from fast_style_transfer import FastStyle

import numpy as np

class Ruder(FastStyle):
  def __init__(self, n_styles=1):
    FastStyle.__init__(self)
    self.method = 'ruder'
    self.train_dir += self.method + '/'
    self.loss_labels = ['total', 'content', 'style', 'temporal_loss']
    self.loss_letters = ["a", "b", "c"]

  def roll(self, p):
    return True if np.random.random() < p else False
  
  def setup_method(self, run_id, emphasis_parameter):
    run_id += self.concat_id(emphasis_parameter)
    
    if run_id[0] == 'm':
      pre_style_path = self.train_dir[:5] + "FC2/dumoulin/msid" + run_id[4] + "_ep20_bs16_lr-3_a0_a0_a0_b1_b1_b1/epoch_19.pth"
      n_styles = int(run_id[4])
    else:
      pre_style_path = self.train_dir[:5] + "FC2/johnson/sid" + run_id[3] + "_ep20_bs16_lr-3_a0_b1_d-4/epoch_19.pth"
      n_styles = 1
    
    self.model = FastStyleNet(3 + 1 + 3, n_styles).to(self.device)
    self.pre_style_model = FastStyleNet(3, n_styles).to(self.device)
    
    self.pre_style_model.load_state_dict(torch.load(pre_style_path))
    self.first_frame = True
    return run_id
  
  def train_method(self, imgs, masks, flows, emphasis_parameter, style_id):
    alpha, beta, gamma = emphasis_parameter
    
    masks = torch.split(masks, 1, dim=1)
    flows = torch.split(flows, 2, dim=1)
    
    rand_roll = self.roll(0.5)
        
    if rand_roll:
      _, styled_img1 = self.pre_style_model(imgs[0], s_id=style_id)
      styled_img1 /= 255.0
      
      warped1 = warp(styled_img1, flows[0])

      _, styled_img2 = self.model(torch.cat((imgs[1], masks[0], warped1), 1), s_id=style_id)
      styled_img2 /= 255.0
      loss_img = imgs[1]
      loss_styled = styled_img2
      loss_warped = warped1
      
      if len(imgs) > 2:
        warped2 = warp(styled_img2, flows[1])
        _, styled_img3 = self.model(torch.cat((imgs[2], masks[1], warped2), 1), s_id=style_id)
        styled_img3 /= 255.0
        loss_img = imgs[2]
        loss_styled = styled_img3
        loss_warped = warped2
        
      if len(imgs) > 4:
        warped3 = warp(styled_img3, flows[2])
        _, styled_img4 = self.model(torch.cat((imgs[3], masks[2], warped3), 1), s_id=style_id)
        styled_img4 /= 255.0
        warped4 = warp(styled_img4, flows[3])
        _, styled_img5 = self.model(torch.cat((imgs[4], masks[3], warped4), 1), s_id=style_id)
        styled_img5 /= 255.0
        loss_img = imgs[4]
        loss_styled = styled_img5
        loss_warped = warped4
        
    else:
      _, styled_img2 = self.model(torch.cat((imgs[1], 0.0*masks[0], 0.0*imgs[1]), 1), s_id=style_id)
      styled_img2 /= 255.0
      loss_img = imgs[1]
      loss_styled = styled_img2
      loss_warped = styled_img2

    styled_features = self.vgg(self.normalize(loss_styled))
    img_features = self.vgg(self.normalize(loss_img))

    content_loss = alpha*self.L2distance(styled_features[2], img_features[2])

    style_loss = 0
    for i, gram_s in enumerate(self.styles[style_id]):
      gram_img1 = self.gram_matrix(styled_features[i])
      style_loss += ((gram_img1 - gram_s)**2).mean()#float(weight)*
      
    style_loss *= beta
    
    if rand_roll:
      temporal_loss = gamma*((masks[-1]*(loss_warped - loss_styled))**2).mean()
    else:
      temporal_loss = 0.0

    loss = content_loss + style_loss + temporal_loss
    
    losses = tuple([loss, style_loss, content_loss, temporal_loss])
    loss_string = " L: %.4f CL: %.4f SL: %.4f TL: %.4f" % losses

    loss.backward()
    
    return losses, styled_img2, loss_string

  def infer_method(self, params, style_id):
    torch_f = params[0]
    
    if params[1] == None:#self.first_frame:
      _, styled_frame = self.pre_style_model(torch_f, s_id=style_id)
      #self.first_frame = False
    else:
      torch_m = params[1]
      torch_w = params[2]#torch.from_numpy(params[2]).to(self.device).permute(2, 0, 1).float().unsqueeze(0)
      _, styled_frame = self.model(torch.cat((torch_f, torch_m, torch_w), 1), s_id=style_id)
    
    return styled_frame/255.0
