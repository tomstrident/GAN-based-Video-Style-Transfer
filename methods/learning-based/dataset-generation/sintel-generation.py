# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 01:10:58 2019

@author: Tom
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:24:51 2019

@author: Tom
"""

import os
import cv2
import imageio
import numpy as np

import tensorflow as tf
from skimage import transform

from network import pyramid_processing
from data_augmentation import flow_resize
from config.extract_config import config_dict
from flowlib import flow_to_color, read_flo
from utils import mvn

'''
fpath = 'D:/Datasets/MPI-Sintel-complete/training/clean/market_2/'
flist = os.listdir(fpath)
flist.sort()

writer = imageio.get_writer(fpath + 'market_2.mp4', fps=10)

for fp in flist:
  im = imageio.imread(fpath + fp)
  writer.append_data(im)

writer.close()


reader = imageio.get_reader(fpath + 'market_2.mp4')
print(reader.count_frames())

blah
'''
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
  
  A_m = A#np.moveaxis(A[0], 0, 2)
  #assert A_m.shape[-3:-1] == flow.shape[-3:-1], "dimension error: input and \
  #                                             flow size do not match"
  h, w = flow.shape[:2]
  x = (flow[...,0] + np.arange(w)).astype(A.dtype)
  y = (flow[...,1] + np.arange(h)[:,np.newaxis]).astype(A.dtype)

  W_m = cv2.remap(A_m, x, y, cv2.INTER_LINEAR)

  return W_m.reshape(A.shape)

#--------------------------------------------------------------------------

#path = 'C:/Users/Tom\Documents/Python Scripts/ReCoNet-PyTorch/vid/vsttest/'
#vid = imageio.get_reader('C:/Users/Tom/Documents/Python Scripts/ReCoNet-PyTorch/vsttest.mp4')

#vid_id = 'alley_2'
#vid_id = 'market_6'
sintel_video = 'temple_2'
out_path = 'F:/Test/' + sintel_video + '/'
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

frame_list.sort()
flow_list.sort()
mask_list.sort()

tuple_len = 3
frame_tuple = []

for i, frame_id in enumerate(frame_list):
  
  frame = imageio.imread(frame_path + frame_id)
  frame = np.float32(frame.reshape((1,) + frame.shape)/255.0)
  frame_tuple.append(frame)
  
  if len(frame_tuple) < tuple_len:
    continue
  elif len(frame_tuple) > tuple_len:
    frame_tuple.pop(0)
  
  #print(i - tuple_len + 1)
  
  img0, img1, img2 = frame_tuple#, img3
  
  fid = (sintel_video + '_%d') % (i - tuple_len + 1)
  #imageio.imwrite(out_path + fid + '.png', img0)
  
  #, cf, cb
  #, flow_fw_color, flow_bw_color
  #ff, bf = sess.run([flow_fw['full_res'], flow_bw['full_res']], feed_dict={b_img0:img0, b_img1:img1, b_img2:img2})
  
  #imageio.imwrite('cf.png', cf[0])
  #imageio.imwrite('cb.png', cb[0])
  #blah
  
  #ff = sess.run([flow_fw['full_res']], feed_dict={b_img0:img0, b_img1:img1, b_img2:img2})[0][0]
  #bf = sess.run([flow_bw['full_res']], feed_dict={b_img0:img1, b_img1:img2, b_img2:img3})[0][0]
  
  ff = read_flo(flow_path + flow_list[i - tuple_len + 1])
  bf = sess.run([flow_bw['full_res']], feed_dict={b_img0:img0, b_img1:img1, b_img2:img2})[0][0]
  
  imageio.imwrite(out_path + fid + '_img0.png', img0[0])
  imageio.imwrite(out_path + fid + '_img1.png', img1[0])
  imageio.imwrite(out_path + fid + '_img2.png', img2[0])
  #imageio.imwrite(out_path + fid + '_img3.png', img3[0])
  
  #print(bf.shape, ff.shape)
  #print(img1[0].shape, img0[0].shape)
  #warp1 = warp_image(img0[0], bf)
  #warp2 = warp_image(img1[0], ff)
  
  #imageio.imwrite(out_path + fid + '_warp1.png', warp1)
  #imageio.imwrite(out_path + fid + '_warp2.png', warp2)
  
  mask = imageio.imread(mask_path + mask_list[i - tuple_len + 1])
  mask = 1.0 - mask/255.0
  imageio.imwrite(out_path + fid + '_mask.png', mask)
  mask = mask.reshape((1,) + mask.shape + (1,))
  
  wf = warp_flow(ff, bf)
  mask2 = fb_check(wf, bf)
  imageio.imwrite(out_path + fid + '_mask_2.png', mask2)
  mask2 = mask2.reshape((1,) + mask2.shape + (1,))
  blah
  
  #print(img0.shape)
  #print(img1.shape)
  #print(mask.shape)
  #print(bf.shape)
  #print(ff[0].shape[-3:-1])
  #print(bf[0].shape[:2])
  #blah
  
  #ff, bf = frame_pair
  #flow_data.append(np.array([ff[0], bf[0]]))
  #np.save(flo_path + avi.split('.')[0] + '_' +  str(sht) + '.npy', np.array(flow_data))
  
  frames = [img0, img1]
  flows = [ff, bf]
  masks = [mask, mask2]
  
  batch_data = np.concatenate((mask, bf), axis=3)
  batch_data = batch_data.astype(np.float32)

  np.save(out_path + fid + '.npy', batch_data)
  print(fid + '.npy')
  














