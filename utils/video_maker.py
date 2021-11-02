# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:47:25 2021

@author: Tom
"""

import os
import imageio

#sintel_eval = "G:/Code/ConGAN/eval/"
#sintel_eval = "G:/Code/CycleGAN/eval/"
#sintel_eval = "G:/Code/LBST/eval_sintel/johnson/"
#sintel_eval = "G:/Code/LBST/eval_sintel/ruder/"
#sintel_eval = "G:/Code/LBST/eval_sintel/huang/"
sintel_eval = "G:/Code/LBST/eval_sintel/dumoulin/"
#sintel_eval = "G:/Code/MoGAN/eval/"
#sintel_eval = "G:/Code/OBST/eval_sintel/0/"
#sintel_eval = "G:/Code/OBST/eval_sintel/2000/"
#sintel_eval = "G:/Code/StarGAN/sintel_eval/"
#sintel_eval = "G:/Code/StarGANv2Adv/expr/sintel_eval/"
#sintel_eval = "G:/Code/StarGANv2AdvCon/expr/sintel_eval/"

video_ids = ["alley_2", "market_6", "temple_2"]
fps = 18

for vid in video_ids:
  for sid in range(1, 4):
    vid_name = vid + ("_s%d" % sid)
    vid_path = sintel_eval + vid_name + "/"
    #print(vid_path)
    #blah

    frame_list = os.listdir(vid_path)
    frame_list.sort()
    
    writer = imageio.get_writer(sintel_eval + vid_name + ".mp4", fps=fps)
    
    for frame in frame_list:
      im = imageio.imread(vid_path + frame)
      writer.append_data(im)
  
    writer.close()