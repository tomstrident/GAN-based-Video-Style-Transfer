# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:50:03 2020

@author: Tom
"""

import cv2
import os
import time
import imageio
#from scipy.ndimage.interpolation import shift, zoom, rotate
import numpy as np

'''
frame = np.float32(imageio.imread('D:/Datasets/train2014/COCO_train2014_000000000009.jpg')/255.0)
imageio.imwrite('F:/Datasets/COCO/DATAFiles/1.png', frame)

rows,cols, _ = frame.shape


tx = 20
ty = 20
phi = 32
s=1.2

T = np.float32([[1,  0, tx],
                [ 0, 1, ty],
                [ 0, 0,  1]])

RS = cv2.getRotationMatrix2D((cols/2,rows/2), phi, s)
RS = np.vstack((RS, np.float32([0, 0, 1])))
TSR = np.matmul(T, RS)

TSR_1 = np.linalg.inv(TSR)

print(TSR)
print(TSR_1)

TSR = TSR[:2]
TSR_1 = TSR_1[:2]



phi = 3.141592*(phi/180)
a = np.cos(phi)
b = np.sin(phi)

T_mat = np.float32([[sx*a, -sx*b, tx],
                    [sy*b,  sy*a, ty],
                    [   0,     0,  1]])
print(T_mat)
blah

print(frame.dtype, T_mat.dtype)

result = cv2.warpAffine(frame, TSR, (cols,rows), flags=cv2.INTER_LINEAR)
imageio.imwrite('F:/Datasets/COCO/DATAFiles/2.png', result)

result2 = cv2.warpAffine(result, TSR_1, (cols,rows), flags=cv2.INTER_LINEAR)
imageio.imwrite('F:/Datasets/COCO/DATAFiles/3.png', result2)

blah
'''

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

def warp_flow(A, flow):
    
  assert A.shape[-3:-1] == flow.shape[-3:-1], "dimension error: input and \
                                               flow size do not match"
  h, w = flow.shape[:2]
  x = (flow[...,0] + np.arange(w)).astype(A.dtype)
  y = (flow[...,1] + np.arange(h)[:,np.newaxis]).astype(A.dtype)
  
  return cv2.remap(A, x, y, cv2.INTER_LINEAR)

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
  #weights[np.where(motion_boundaries)] = 0
  
  return weights

class DatasetGenerator():
  def __init__(self, 
               batch_size=(256, 256),
               input_path='D:/Datasets/train2014/', 
               output_path='D:/Datasets/CO5/'):

    self.in_path = input_path
    self.out_path = output_path
    
    self.batch_size = batch_size
    self.batch_dims = np.int32(np.array(batch_size)/2)
    
    self.data_list = os.listdir(input_path)
    self.data_list.sort()
    self.data_list = self.data_list[:9627]
    self.data_len = len(self.data_list)
    
    self.transforms = []
  
  def find_center(self, img_shape):
    return np.int32(np.round(np.array(img_shape[:2])/2)) #could produce errors
  
  def crop_batch(self, data):
    bh, bw = self.batch_dims
    ch, cw = self.find_center(data.shape)
    #print(ch-bh, ch+bh, cw-bw, cw+bw, data.shape)
    return data[ch-bh:ch+bh,cw-bw:cw+bw,:]
  
  def create_flowgrid(self, img_shape):
    h, w, _ = img_shape
    
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    
    return np.moveaxis(np.array(np.meshgrid(x, y)), 0, -1)
  
  def generate_tsr(self, img_size, pmin=-32, pmax=32):
    shift_y, shift_x, rot = np.random.randint(low=pmin, high=pmax, size=3)
    
    pix_range = np.arange(pmin, pmax+2, 2)
    #zx, zy = np.random.choice(pix_range, 2)
    scal = np.random.choice(pix_range, 1)
  
    rows, cols, _ = img_size
    size = min(rows, cols)
    scal = (size + scal)/size
    
    #zoom_x = (img_size[1] + zx)/img_size[1]
    #zoom_y = (img_size[0] + zy)/img_size[0]
    
    T = np.float32([[1,  0, shift_x],
                    [ 0, 1, shift_y],
                    [ 0, 0,  1]])

    RS = cv2.getRotationMatrix2D((cols/2,rows/2), rot, scal)
    RS = np.vstack((RS, np.float32([0, 0, 1])))
    TSR = np.matmul(T, RS)
    
    return TSR#np.array([shift_x, shift_y, rot, zoom_x, zoom_y])
  '''
  def transform(self, data, params):
    shift_x, shift_y, rot, zoom_x, zoom_y = params
    
    alt_data = data
    for idx, t in enumerate(self.transforms):
      alt_data = t(params[i])
    alt_data = shift(alt_data, [shift_y, shift_x, 0]) #ok
    alt_data = rotate(alt_data, rot, reshape=False) #ok
    alt_data = zoom(alt_data, [zoom_y, zoom_x, 1])
    
    return alt_data
  '''
  def transform(self, data, tsr):
    #shift_x, shift_y, rot, zoom_x, zoom_y = params
    
    alt_data = cv2.warpAffine(data, tsr[:2], data.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    #alt_data = shift(data, [shift_y, shift_x, 0])
    #alt_data = rotate(alt_data, rot, reshape=False)
    #alt_data = zoom(alt_data, [zoom_y, zoom_x, 1])
    
    return alt_data
  
  def inv_transform(self, data, tsr):
    tsr_1 = np.linalg.inv(tsr)
    #shift_x, shift_y, rot = -params[:3]
    #zoom_x, zoom_y = 1/params[3:]
    
    alt_data = cv2.warpAffine(data, tsr_1[:2], data.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    #alt_data = zoom(data, [zoom_y, zoom_x, 1])
    #alt_data = rotate(alt_data, rot, reshape=False)
    #alt_data = shift(alt_data, [shift_y, shift_x, 0])

    return alt_data
  
  def compute_flow(self, img_shape, params):
    grid = self.create_flowgrid(img_shape)
    
    fw_grid = self.inv_transform(grid, params)
    bw_grid = self.transform(grid, params)
    
    fw_grid = self.crop_batch(fw_grid)
    bw_grid = self.crop_batch(bw_grid)
    batch_grid = self.crop_batch(grid)

    ff = np.float32(fw_grid - batch_grid)
    bf = np.float32(bw_grid - batch_grid)
    
    return ff, bf
  
  def sanity_check(self, gt_image, w_image, w_mask):
    return ((w_mask*(gt_image - w_image))**2).mean()
  
  def generate(self, tuple_len=3, debug=False):
    N_means = 10
    mov_mean = np.zeros((N_means,))
    sanity_values = []
    
    for idx, img_path in enumerate(self.data_list):
      
      #if idx < 2580:
      #  continue
      
      t_start = time.time()
      sanity_value = 0.0
      
      frames = []
      flows = []
      masks = []
      
      frame = np.float32(imageio.imread(self.in_path + img_path)/255.0)
      
      if frame.shape[0] < self.batch_size[0] or frame.shape[1] < self.batch_size[1]:
        h, w = np.maximum(frame.shape[:2], self.batch_size)
        frame = cv2.resize(frame, (w, h));
      
      if len(frame.shape) == 2:#grayscale
        frame = np.moveaxis(np.array([frame, frame, frame]), 0, -1)
      first_frame = frame.copy()
      frames.append(self.crop_batch(frame))
      #params = np.array([0, 0, 0, 1, 1])
      if debug:
        imageio.imwrite(self.out_path + 'frame0.png', frames[-1])
      #print(frame.min(), frame.max())
      
      mat_vec = []
      
      for i in range(tuple_len-1):
        #params = self.generate_params(frame.shape)
        tsr_mat = self.generate_tsr(frame.shape)
        #test_mat *= tsr_mat
        mat_vec.append(tsr_mat)
        
        frame = self.transform(frame, tsr_mat)
        frame = np.clip(frame, 0.0, 1.0)

        ff, bf = self.compute_flow(frame.shape, tsr_mat)
        wf = warp_flow(ff, bf)
        mask = fb_check(wf, bf)
        mask = mask.reshape(mask.shape + (1,))
        #imageio.imwrite(self.out_path + 'mask' + str(i) + '_1.png', mask)
        #wf = warp_flow(bf, ff)
        #mask2 = fb_check(wf, ff)
        #imageio.imwrite(self.out_path + 'mask' + str(i) + '_2.png', mask2)
        
        frames.append(self.crop_batch(frame))
        
        if debug:
          imageio.imwrite(self.out_path + 'frame' + str(i+1) + '.png', frames[-1])
        
        warp1 = warp_image(frames[-2], bf)
        #warp2 = warp_image(frames[-1], ff)

        #mask2 = mask2.reshape(mask2.shape + (1,))
        #test_loss1 = temp_loss(frames[-1], warp1, mask1)
        #test_loss2 = temp_loss(frames[-2], warp2, mask2)
        #print(test_loss1, test_loss2)
        
        #imageio.imwrite(self.out_path + 'warp' + str(i+1) + '_1.png', warp1)
        #imageio.imwrite(self.out_path + 'warp' + str(i+1) + '_2.png', warp2)
        
        flows.append(bf)
        masks.append(mask)
        
        sanity_value += self.sanity_check(frames[-1], warp1, mask)
      
      flow1 = flows[0]
      flow2 = flows[1]
      
      test_flow = warp_flow(flow1, flow2)
      warp1 = warp_image(frames[0], test_flow)
      
      imageio.imwrite(self.out_path + 'frame2_orig.png', frames[2])
      imageio.imwrite(self.out_path + 'frame0_warped.png', warp1)
      
      blah
      
      tsr_51 = np.identity(3)
      for m in mat_vec:#[::-1]
        tsr_51 = np.matmul(m, tsr_51)
        #tsr_51 = np.matmul(tsr_51, m)
      
      tsr_51 = np.linalg.inv(tsr_51)
      
      #test_frame2 = self.transform(frame, tsr_51)
      #imageio.imwrite(self.out_path + 'frame4_warped.png', test_frame2)
      #imageio.imwrite(self.out_path + 'frame0_orig.png', first_frame)
      #blah
      
      ff, bf = self.compute_flow(first_frame.shape, tsr_51)
      wf = warp_flow(ff, bf)
      mask = fb_check(wf, bf)
      mask = mask.reshape(mask.shape + (1,))
      #imageio.imwrite(self.out_path + 'mask' + str(i) + '_1.png', mask)
      warp1 = warp_image(frames[-1], bf)
      #sanity_value += self.sanity_check(frames[0], warp1, mask)
      #imageio.imwrite(self.out_path + 'frame0_orig.png', frames[0])
      #imageio.imwrite(self.out_path + 'frame4_warped.png', warp1)
      #blah

      flows.append(bf)
      masks.append(mask)
      
      frames = np.array(frames)
      flows = np.array(flows)
      masks = np.array(masks)
      #print(frames.shape)
      #print(flows.shape)
      #print(masks.shape)
      
      if frames.shape[1:3] != self.batch_size:
        print(frames.shape, idx)
        
      assert frames.shape[1:3] == self.batch_size
      assert flows.shape[1:3] == self.batch_size
      assert masks.shape[1:3] == self.batch_size
      
      #assert sanity_value > 0
      
      #batch_data = np.array([frames, flows, masks])
      batch_data = np.concatenate([frames, flows, masks], axis=3)
      
      np.save(self.out_path + 'DATAFiles/' + img_path[:-4] + '.npy', batch_data)
    
      t_end = time.time()
      mov_mean[idx % N_means] = t_end - t_start
      mean = mov_mean.mean()
      
      sanity_values.append(sanity_value)
      
      print('(%d/%d) %.4f' % (idx, self.data_len, sanity_value), img_path, 'ETA: %.4f' % ((mean*(self.data_len - idx))/60))

    sanity_values = np.array(sanity_values)
    np.save(self.out_path + '_sanity.npy', sanity_values)

def main():
  dset_generator = DatasetGenerator()
  dset_generator.generate()
  

if __name__ == '__main__':
  main()
