# -*- coding: utf-8 -*-

import torch
from fs_lib import warp
from network import FastStyleNet
from fast_style_transfer import FastStyle

class Huang(FastStyle):
  def __init__(self):
    FastStyle.__init__(self)
    self.method = 'huang'
    self.train_dir += self.method + '/'
    self.loss_labels = ['total', 'content', 'style', 'temporal_loss', 'tv']
    self.loss_letters = ["a", "b", "c", "d"]
    
  def setup_method(self, run_id, emphasis_parameter):
    run_id += self.concat_id(emphasis_parameter)
    
    if run_id[0] == 'm':
      n_styles = int(run_id[4])
    else:
      n_styles = 1
    
    self.model = FastStyleNet(3, n_styles).to(self.device)
    
    return run_id

  def train_method(self, imgs, masks, flows, emphasis_parameter, style_id):
    alpha, beta, gamma, delta = emphasis_parameter

    masks = torch.split(masks, 1, dim=1)
    flows = torch.split(flows, 2, dim=1)

    _, styled_img1 = self.model(imgs[0], s_id=style_id)
    styled_img1 /= 255.0
    _, styled_img2 = self.model(imgs[1], s_id=style_id)
    styled_img2 /= 255.0

    styled_features1 = self.vgg(self.normalize(styled_img1))
    styled_features2 = self.vgg(self.normalize(styled_img2))
    img_features1 = self.vgg(self.normalize(imgs[0]))
    img_features2 = self.vgg(self.normalize(imgs[1]))

    content_loss = (alpha/2)*(self.L2distance(styled_features1[2], img_features1[2]) + self.L2distance(styled_features2[2], img_features2[2]))

    style_loss = 0
    for i, gram_s in enumerate(self.styles[style_id]):
      gram_img1 = self.gram_matrix(styled_features1[i])
      gram_img2 = self.gram_matrix(styled_features2[i])
      style_loss += ((gram_img1 - gram_s)**2).mean()#float(weight)*
      style_loss += ((gram_img2 - gram_s)**2).mean()#float(weight)*
      
    style_loss *= (beta/2)

    warped = warp(styled_img1, flows[0])
    temporal_loss = gamma*(((masks[0]*(styled_img2 - warped))**2).mean())# + ((masks[0]*(styled_img2 - warped))**2).mean()

    tv_loss = delta*self.calc_tv_loss(styled_img1)

    loss = content_loss + style_loss + temporal_loss + tv_loss

    losses = tuple([loss, content_loss, style_loss, temporal_loss, tv_loss])
    loss_string = " L: %.4f CL: %.4f SL: %.4f  TL: %.4f RL: %.4f" % losses

    loss.backward()
    
    return losses, styled_img2, loss_string
    
  def infer_method(self, params, style_id):
    _, styled_frame = self.model(params[0], s_id=style_id)
    return styled_frame/255.0
