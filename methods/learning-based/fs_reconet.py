# -*- coding: utf-8 -*-

import torch
from fs_lib import warp
from network import FastStyleNet
from fast_style_transfer import FastStyle

class Reconet(FastStyle):
  def __init__(self):
    FastStyle.__init__(self)
    self.method = 'reconet'
    self.train_dir += self.method + '/'
    self.loss_labels = ['total', 'content', 'style', 'f_temporal_loss', 'o_temporal_loss', 'tv']
    self.loss_letters = ["a", "b", "cf", "co", "d"]
  
  def setup_method(self, run_id, emphasis_parameter):
    run_id += self.concat_id(emphasis_parameter)
    
    if run_id[0] == 'm':
      n_styles = int(run_id[4])
    else:
      n_styles = 1
    
    self.model = FastStyleNet(3, n_styles).to(self.device)
    
    return run_id

  def train_method(self, imgs, masks, flows, emphasis_parameter, style_id):
    alpha, beta, gamma_f, gamma_o, delta = emphasis_parameter

    masks = torch.split(masks, 1, dim=1)
    flows = torch.split(flows, 2, dim=1)
    
    feature_map1, styled_img1 = self.model(imgs[0], s_id=style_id)
    styled_img1 /= 255.0
    feature_map2, styled_img2 = self.model(imgs[1], s_id=style_id)
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

    tv_loss = (delta/2)*(self.calc_tv_loss(styled_img1) + self.calc_tv_loss(styled_img2))

    feature_flow = torch.nn.functional.interpolate(flows[0], size=feature_map1.shape[2:], mode='bilinear')
    feature_flow[:, 0, :, :] *= float(feature_map1.shape[2])/flows[0].shape[2]
    feature_flow[:, 1, :, :] *= float(feature_map1.shape[3])/flows[0].shape[3]
    feature_mask = torch.nn.functional.interpolate(masks[0], size=feature_map1.shape[2:], mode='bilinear')
    warped_fmap = warp(feature_map1, feature_flow)
    
    f_temporal_loss = gamma_f*(((feature_mask*(feature_map2 - warped_fmap))**2).mean())

    output_term = styled_img2 - warp(styled_img1, flows[0])
    input_term = imgs[1] - warp(imgs[0], flows[0])
    input_term = (0.2126*input_term[:, 0, :, :] + 0.7152*input_term[:, 1, :, :] + 0.0722*input_term[:, 2, :, :]).unsqueeze(1)
    
    o_temporal_loss = gamma_o*(((masks[0]*(output_term - input_term))**2).mean())

    loss = content_loss + style_loss + f_temporal_loss + o_temporal_loss + tv_loss
    
    losses = tuple([loss, content_loss, style_loss, f_temporal_loss, o_temporal_loss, tv_loss])
    loss_string = " L: %.4f CL: %.4f SL: %.4f FTL: %.4f OTL: %.4f TV: %.4f" % losses
    
    loss.backward()
    
    return losses, styled_img2, loss_string

  def infer_method(self, params, style_id):
    _, styled_frame = self.model(params[0], s_id=style_id)
    return styled_frame/255.0
    