# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:24:51 2019

@author: Tom
"""

import time
import os
import cv2
import imageio
import numpy as np
import tensorflow as tf
from skimage import transform

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
  
''''''

#==============================================================================

path = 'D:/Datasets/Hollywood2-actions/Hollywood2/'

avi_path = path + 'AVIClips/'
avi_list = os.listdir(avi_path)
avi_list.sort()

shot_path = path + 'ShotBounds/'
shot_list = os.listdir(shot_path)
shot_list.sort()

''''''
batch_len = 0
for i, shot in enumerate(shot_list):
  batch_len += 1
  
  with open(shot_path + shot, 'r') as shot_file:
    shot_data = shot_file.read()
  
  shot_bounds = shot_data.split(' ')[:-1]
    
  if len(shot_bounds) > 1:
    batch_len += len(shot_bounds)

print('Hollywood2 dataset:')
print('  number of videos:', len(avi_list))
print('  number of shots:', batch_len)

#data_path = path + 'DATAFiles/'
ssd_path = 'D:/Datasets/HW5/'
data_path = ssd_path + 'DATAFiles/'

if not os.path.exists(data_path):
  os.makedirs(data_path)

tuple_len = 5
min_shot_len = tuple_len + 2
batch_size = (256, 256)
h, w = (np.array(batch_size)/2).astype(np.int32)

N_means = 10
mov_mean = np.zeros((N_means,))
batch_count = 0

for idx, avi in enumerate(avi_list):
  if idx < 1:
    continue
  #if idx < 1950:
  #  continue
  
  avi_reader = imageio.get_reader(avi_path + avi)
  last_fid = avi_reader.count_frames()
  #print(avi)
  #print(shot_list[idx])
  
  with open(shot_path + shot_list[idx], 'r') as shot_file:
    shot_data = shot_file.read()
    shot_bounds = shot_data.split(' ')[:-1]
  
  #print(shot_bounds)
  
  if not shot_bounds[0]:
    shot_bounds = [last_fid]
  else:
    shot_bounds = [int(sb) for sb in shot_bounds] + [last_fid]
  
  shot_bounds = np.array([0] + shot_bounds)
  #print(shot_bounds)
  
  for i, shot in enumerate(shot_bounds[1:]):
    t_start = time.time()
    #print(shot - shot_bounds[i])
    if shot - shot_bounds[i] < min_shot_len:
      continue
    
    step_size = (shot - shot_bounds[i])/min_shot_len
    #print(step_size)
    
    #--------------------------------------------------------------------------
    
    frames = []
    flows = []
    masks = []
    
    for t_idx in range(min_shot_len):
      f_idx = shot_bounds[i] + int(t_idx*step_size)
      frame = avi_reader.get_data(f_idx)
      frames.append(frame)
      #imageio.imwrite(data_path + str(t_idx) + '.jpg', frame)
      
    ih, iw = np.floor(np.array(frames[0].shape[0:2])/2).astype(np.int32)
    
    if ih < h or iw < w:
      ih, iw = (max(ih, h), max(iw, w))
      frames = [255.0*transform.resize(img, (2*ih, 2*iw)) for img in frames]
    
    frames = [np.float32(img.reshape((1,) + img.shape)/255.0) for img in frames]
    
    for f_idx in range(tuple_len-1):
      img0, img1, img2, img3 = frames[f_idx:f_idx + 4]
      #fc, bc = sess.run([flow_fw_color, flow_bw_color], feed_dict={b_img0:img0, b_img1:img1, b_img2:img2})
      #imageio.imwrite(data_path + str(f_idx) + '_ff.png', fc[0])
      #imageio.imwrite(data_path + str(f_idx) + '_fb.png', bc[0])
      #print(img0.shape, img1.shape, img2.shape, img3.shape)
      ff = sess.run([flow_fw['full_res']], feed_dict={b_img0:img0, b_img1:img1, b_img2:img2})[0]
      bf = sess.run([flow_bw['full_res']], feed_dict={b_img0:img1, b_img1:img2, b_img2:img3})[0]
      
      #print(ff[0].shape, bf[0].shape)
      #print(img0[0].dtype, bf[0].dtype)
      #test_warp = warp_image(img2[0], ff[0])
      #test_warp2 = warp_image(img1[0], bf[0])
      
      wf = warp_flow(ff[0], bf[0])
      mask = fb_check(wf, bf[0])
      #imageio.imwrite(data_path + str(f_idx) + '_mask.png', mask)
      #imageio.imwrite(data_path + str(f_idx) + '_warp1.png', test_warp)
      #imageio.imwrite(data_path + str(f_idx) + '_warp2.png', test_warp2)
      
      mask = mask.reshape((1,) + mask.shape + (1,))
      flows.append(bf)
      masks.append(np.float32(mask))
    '''
    #circle flow
    img0, img1, img2, img3 = frames[-3:-1] + frames[1:3]
    imageio.imwrite(data_path + 'test_img0.png', img0[0])
    imageio.imwrite(data_path + 'test_img1.png', img1[0])
    imageio.imwrite(data_path + 'test_img2.png', img2[0])
    imageio.imwrite(data_path + 'test_img3.png', img3[0])
    ff = sess.run([flow_fw['full_res']], feed_dict={b_img0:img0, b_img1:img1, b_img2:img2})[0]
    bf = sess.run([flow_bw['full_res']], feed_dict={b_img0:img1, b_img1:img2, b_img2:img3})[0]
    test_warp = warp_image(img2[0], ff[0])
    test_warp2 = warp_image(img1[0], bf[0])
    wf = warp_flow(ff[0], bf[0])
    mask = fb_check(wf, bf[0])
    imageio.imwrite(data_path + 'test_mask.png', mask)
    imageio.imwrite(data_path + 'test_warp1.png', test_warp)
    imageio.imwrite(data_path + 'test_warp2.png', test_warp2)
    blah
    mask = mask.reshape((1,) + mask.shape + (1,))
    flows.append(bf)
    masks.append(np.float32(mask))
      
    blah
    '''
    frames = np.array(frames[1:-1])[:,0,ih-h:ih+h,iw-w:iw+w,:]
    flows = np.array(flows)[:,0,ih-h:ih+h,iw-w:iw+w,:]
    masks = np.array(masks)[:,0,ih-h:ih+h,iw-w:iw+w,:]
    
    #frames = np.concatenate(frames, axis=2)
    #flows = np.concatenate(flows, axis=2)
    #masks = np.concatenate(masks, axis=2)
    
    #print(frames.shape, frames.dtype)
    #print(flows.shape, flows.dtype)
    #print(masks.shape, masks.dtype)
    
    #batch_data = np.concatenate(frames + masks + flows, axis=3)
    batch_data = np.array([frames, flows, masks])
    #batch_data = batch_data[:,ih-h:ih+h,iw-w:iw+w,:]
    #print(batch_data.shape, batch_data.dtype)

    np.save(data_path + avi.split('.')[0] + '_' +  str(i) + '.npy', batch_data)
  
    t_end = time.time()
    mov_mean[idx % N_means] = t_end - t_start
    mean = mov_mean.mean()
    batch_count += 1
    
    print('%d/%d' % (batch_count, batch_len), avi.split('.')[0] + '_' + str(i) + '.npy', 'ETA', ((batch_len - batch_count)*mean)/60)

#==============================================================================

