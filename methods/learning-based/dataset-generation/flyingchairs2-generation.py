# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:15:53 2020

@author: Tom
"""

import os
import cv2
import imageio
import numpy as np
'''
import tensorflow as tf
from network import pyramid_processing
from data_augmentation import flow_resize
from config.extract_config import config_dict
from flowlib import flow_to_color
from utils import mvn

#--------------------------------------------------------------------------

config = config_dict('./config/config.ini')
restore_model = config['test']['restore_model']

b_img0 = tf.placeholder(tf.float32, shape=(1, None, None, 3))
b_img1 = tf.placeholder(tf.float32, shape=(1, None, None, 3))
b_img2 = tf.placeholder(tf.float32, shape=(1, None, None, 3))

#b_img0 = tf.placeholder(tf.float32, shape=(1, 384, 512, 3))
#b_img1 = tf.placeholder(tf.float32, shape=(1, 384, 512, 3))
#b_img2 = tf.placeholder(tf.float32, shape=(1, 384, 512, 3))

b_img0 = mvn(b_img0)
b_img1 = mvn(b_img1)
b_img2 = mvn(b_img2)

img_shape = tf.shape(b_img0)
h = img_shape[1]
w = img_shape[2]

new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)

batch_img0 = tf.image.resize_images(b_img0, [new_h, new_w], method=1, align_corners=True)
batch_img1 = tf.image.resize_images(b_img1, [new_h, new_w], method=1, align_corners=True)
batch_img2 = tf.image.resize_images(b_img2, [new_h, new_w], method=1, align_corners=True)

flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=True) 
flow_fw['full_res'] = flow_resize(flow_fw['full_res'], [h, w], method=1)
flow_bw['full_res'] = flow_resize(flow_bw['full_res'], [h, w], method=1)

flow_fw_color = flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
flow_bw_color = flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)

restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
saver = tf.train.Saver(var_list=restore_vars)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, restore_model)

def fb_check(w_warp, w_back): #optimizable? cuda?
    
  weights = np.ones(w_warp.shape[:2])
  
  norm_wb = np.linalg.norm(w_warp + w_back, axis=2)**2.0
  norm_w = np.linalg.norm(w_warp, axis=2)**2.0
  norm_b = np.linalg.norm(w_back, axis=2)**2.0
  
  disoccluded_regions = norm_wb > 0.01*(norm_w + norm_b) + 0.5
  
  norm_u = np.linalg.norm(np.gradient(w_back[...,0]), axis=0)**2.0
  norm_v = np.linalg.norm(np.gradient(w_back[...,1]), axis=0)**2.0
  
  motion_boundaries = norm_u + norm_v > 0.01*norm_b + 0.002
  
  weights[np.where(disoccluded_regions)] = 0
  weights[np.where(motion_boundaries)] = 0
  
  return weights
  
def warp_flow(A, flow):
    
  assert A.shape[-3:-1] == flow.shape[-3:-1], "dimension error: input and \
                                               flow size do not match"
  h, w = flow.shape[:2]
  x = (flow[...,0] + np.arange(w)).astype(A.dtype)
  y = (flow[...,1] + np.arange(h)[:,np.newaxis]).astype(A.dtype)
  
  return cv2.remap(A, x, y, cv2.INTER_LINEAR)

def warp_image(A, flow):
  
  #A_m = np.moveaxis(A, 0, 2)
  #flow = np.moveaxis(flow, 0, 2)
  A_m = A
  
  #print('A', A.shape)
  #print('flow', flow.shape)
  
  #print(A_m.shape[-3:-1], flow.shape[-3:-1])
  #print(A_m.dtype, flow.dtype)
  assert A_m.shape[-3:-1] == flow.shape[-3:-1], "dimension error: input and flow size do not match"
  h, w = flow.shape[:2]
  x = (flow[...,0] + np.arange(w)).astype(A.dtype)
  y = (flow[...,1] + np.arange(h)[:,np.newaxis]).astype(A.dtype)

  W_m = cv2.remap(A_m, x, y, cv2.INTER_LINEAR)

  return W_m.reshape(A.shape)
'''
#==============================================================================

#sintel_video = 'alley_2'
#sintel_video = 'market_6'
sintel_video = 'temple_2'
out_path = 'D:/Datasets/FC5/' + sintel_video + '/'
sintel_path = 'D:/Datasets/MPI-Sintel-complete/training/'
    
frame_path = sintel_path + 'final/' + sintel_video + '/'
flow_path = sintel_path + 'flow/' + sintel_video + '/'
mask_path = sintel_path + 'occlusions/' + sintel_video + '/'

if not os.path.exists(out_path):
  os.makedirs(out_path)

if not os.path.exists(frame_path):
  print('invalid path')

frame_list = os.listdir(frame_path)
flow_list = os.listdir(flow_path)
mask_list = os.listdir(mask_path)

frame_list.sort(reverse=True)
flow_list.sort(reverse=True)
mask_list.sort(reverse=True)

long_term_length = 5

for i in range(len(frame_list)):

  if i - long_term_length - 1 < 0:
    continue
  #print(frame_list[i])
  #blah
  img3 = imageio.imread(frame_path + frame_list[i])
  img2 = imageio.imread(frame_path + frame_list[i-1])
  img1 = imageio.imread(frame_path + frame_list[i-long_term_length])
  img0 = imageio.imread(frame_path + frame_list[i-long_term_length-1])
  print(frame_list[i-long_term_length-1], i-long_term_length-1)#49
  print(frame_list[i-long_term_length], i-long_term_length)#48
  print(frame_list[i-1], i-1)
  print(frame_list[i], i)
  blah

  #imageio.imwrite(out_path + 'img0.png', img0)
  #imageio.imwrite(out_path + 'img1.png', img1)
  #imageio.imwrite(out_path + 'img2.png', img2)
  #imageio.imwrite(out_path + 'img3.png', img3)
  
  img0 = np.float32(img0.reshape((1,) + img0.shape)/255.0)
  img1 = np.float32(img1.reshape((1,) + img1.shape)/255.0)
  img2 = np.float32(img2.reshape((1,) + img2.shape)/255.0)
  img3 = np.float32(img3.reshape((1,) + img3.shape)/255.0)
  
  ff = sess.run([flow_fw['full_res']], feed_dict={b_img0:img0, b_img1:img1, b_img2:img2})[0]
  bf = sess.run([flow_bw['full_res']], feed_dict={b_img0:img1, b_img1:img2, b_img2:img3})[0]
  
  wf = warp_flow(ff[0], bf[0])
  mask = fb_check(wf, bf[0])
  
  #test_warp = warp_image(img2[0], ff[0])
  #test_warp2 = warp_image(img1[0], bf[0])

  #imageio.imwrite(out_path + 'test_mask.png', mask)
  #imageio.imwrite(out_path + 'test_warp1.png', test_warp)
  #imageio.imwrite(out_path + 'test_warp2.png', test_warp2)
  
  mask = mask.reshape((1,) + mask.shape + (1,))

  batch_data = np.concatenate([bf, mask], axis=3)
  
  print(out_path + frame_list[i][:10] + '.npy')
  #blah
  np.save(out_path + frame_list[i][:10] + '.npy', batch_data)
  
  
  
  
  
  
  
  
  
  