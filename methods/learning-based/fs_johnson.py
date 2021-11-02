# -*- coding: utf-8 -*-

from network import FastStyleNet
from fast_style_transfer import FastStyle

class Johnson(FastStyle):
  def __init__(self):
    FastStyle.__init__(self)
    self.method = 'johnson'
    self.train_dir += self.method + '/'
    self.loss_labels = ['total', 'content', 'style', 'tv']
    self.loss_letters = ["a", "b", "d"]

  def setup_method(self, run_id, emphasis_parameter):
    run_id += self.concat_id(emphasis_parameter)
    
    if run_id[0] == 'm':
      n_styles = int(run_id[4])
    else:
      n_styles = 1
      
    self.model = FastStyleNet(3, n_styles).to(self.device)

    return run_id

  def train_method(self, imgs, masks, flows, emphasis_parameter, style_id):
    alpha, beta, delta = emphasis_parameter
    
    _, styled_img1 = self.model(imgs[0])
    styled_img1 /= 255.0
    
    loss_img = imgs[0]
    loss_styled = styled_img1

    styled_features = self.vgg(self.normalize(loss_styled))
    img_features = self.vgg(self.normalize(loss_img))

    content_loss = alpha*self.L2distance(styled_features[2], img_features[2])

    style_loss = 0
    for i, gram_s in enumerate(self.styles[0]):
      gram_img1 = self.gram_matrix(styled_features[i])
      style_loss += ((gram_img1 - gram_s)**2).mean()
      
    style_loss *= beta
    
    tv_loss = delta*self.calc_tv_loss(styled_img1)

    loss = content_loss + style_loss + tv_loss
    
    losses = tuple([loss, content_loss, style_loss, tv_loss])
    loss_string = " L: %.4f CL: %.4f SL: %.4f TV: %.4f" % losses

    loss.backward()
    
    return losses, styled_img1, loss_string

  def infer_method(self, params, style_id):
    _, styled_frame = self.model(params[0], style_id)
    return styled_frame/255.0
