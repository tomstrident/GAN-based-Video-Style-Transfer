# -*- coding: utf-8 -*-

import numpy as np

from fs_johnson import Johnson
from fs_ruder import Ruder
from fs_huang import Huang
from fs_reconet import Reconet
from fs_dumoulin import Dumoulin
import torch

def train_net(setup,
              sid=2, epochs=3, dset='FC2'):
  fs_test, train_params = setup
  fs_test.train(sid, epochs, train_params, dset=dset)

def infer_test(setup,
               sid=2, n_styles=1, epochs=3, n_epochs=2, 
               batchsize=16, learning_rate=1e-3,
               dset='FC2', sintel_id='temple_2', sintel_path='D:/Datasets/', 
               vid_fps=20, out_img_path=None, out_img_num=[10]):
  
  fs_test, infer_params = setup
  ret = fs_test.infer(sid, n_styles, epochs, n_epochs, infer_params, 
                       batchsize, learning_rate,
                       dset, sintel_id, sintel_path,
                       vid_fps, out_img_path, out_img_num)
  return ret

def eval_test(setup, args,
              n_styles=1, epochs=3, n_epochs=2, 
              batchsize=16, learning_rate=1e-3):
  with torch.no_grad():
    fs_test, infer_params = setup
    #fs_test.evaluate_fc2(args, n_styles, epochs, n_epochs, infer_params)
    fs_test.evaluate_sintel(args, n_styles, epochs, n_epochs, infer_params)

def param_var(setup, pos, params, out_path):
  cst = []
  fs_test, std_params = setup
  for p in params:
    var_params = std_params.copy()
    var_params[pos] = p
    train_net([fs_test, var_params], sid=2, epochs=1, dset='FC2')
    cst.append(infer_test([fs_test, var_params], sid=2, n_styles=1, epochs=1, n_epochs=0, sintel_id='temple_2', out_img_path=out_path)[2:])
  cst = np.array(cst)
  cst = np.hstack((cst[:,0], cst[:,1]))
  latex_string = " & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f" % tuple(cst)
  return latex_string

def select_method(method_id):
  if method_id == "Johnson":
    fs_test = Johnson()
    std_params = [1e0, 1e1, 1e-4]
  elif method_id == "Ruder":
    fs_test = Ruder()
    std_params = [1e0, 1e1, 1e2]
    #std_params = [[1e-1, 1e0, 1e1], 1e1, 1e2]
  elif method_id == "Huang":
    fs_test = Huang()
    std_params = [1e0, 1e1, 1e2, 1e-4]
  elif method_id == "ReCoNet":
    fs_test = Reconet()
    std_params = [1e0, 1e1, 1e2, 1e2, 1e-4]
  elif method_id == "Dumoulin":
    fs_test = Dumoulin()
    std_params = [1e0, 1e1]
  else:
    print("invalid method id")
    assert False
    
  return fs_test, std_params

def main():
  alphas = [1e-1, 1e0, 1e1]
  betas =  [1e0, 1e1, 1e2]
  gammas_o = [1e1, 1e2, 1e3]
  gammas_f = [1e2, 1e3, 1e4]#1e0, 1e1, 
  gammas_l = [1e2, 1e3, 1e4]#1e0, 1e1, 
  deltas = [1e-3, 1e-4, 1e-5]
  
  method = "Huang"
  setup = select_method(method)

  train_net(setup, sid=2, epochs=4)
  infer_test(setup, sid=2, n_styles=1)
  
  '''
  #intro
  path = "C:/Users/Tom/Documents/Texmaker/MastersProject/figures/intro/"
  train_net(setup, sid=5)
  infer_test(setup, sid=5, n_styles=1, out_img_path=path, out_img_num=[0, 14, 27])
  
  #4.3.2 Consistency between methods
  cst_vals = []

  cst_vals.append(infer_test(setup, sid=0, sintel_id='temple_2')[2:])
  cst_vals.append(infer_test(setup, sid=1, sintel_id='temple_2')[2:])
  cst_vals.append(infer_test(setup, sid=2, sintel_id='temple_2')[2:])
  cst_vals.append(infer_test(setup, sid=2, sintel_id='market_6')[2:])
  cst_vals.append(infer_test(setup, sid=2, sintel_id='alley_2')[2:])
  
  cst_vals = np.array(cst_vals)
  cst_vals = np.hstack((cst_vals[:,0], cst_vals[:,1]))
  
  latex_string = " & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f" % tuple(cst_vals)
  print(method + latex_string)
  
  blah
  
  #4.3.3 Styling differences
  out_path = 'C:/Users/Tom\Documents/Texmaker/MastersProject/figures/style_comp/ruder/'
  infer_test(sid=0, vid='temple_2', out_img_path=out_path)
  infer_test(sid=1, vid='temple_2', out_img_path=out_path)
  infer_test(sid=2, vid='temple_2', out_img_path=out_path)

  #4.3.4 Computation times
  avg_vals = []
  
  avg_vals.append(infer_test(sid=2, vid='alley_2')[:2])
  avg_vals.append(infer_test(sid=2, vid='market_6')[:2])
  avg_vals.append(infer_test(sid=2, vid='temple_2')[:2])
  avg_vals = np.array(avg_vals)
  avg_vals = np.hstack((avg_vals[:,0], avg_vals[:,1]))
  
  print('Ruder  & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f' % tuple(avg_vals))

  #4.4.1 Emphasis parameter influence
  out_path = "C:/Users/Tom/Documents/Texmaker/MastersProject/figures/param_var/" + method.lower() + "/"
  #out_path = None
  
  a_string = param_var(setup, 0, alphas, out_path)
  b_string = param_var(setup, 1, betas, out_path)
  
  if method == "Johnson":
    d_string = param_var(setup, 2, deltas, out_path)
    print(a_string)
    print(b_string)
    print(d_string)
  
  if method == "Ruder":
    c_string = param_var(setup, 2, gammas_o, out_path)
    print(a_string)
    print(b_string)
    print(c_string)
   
  if method == "Huang":
    c_string = param_var(setup, 2, gammas_o, out_path)
    d_string = param_var(setup, 3, deltas, out_path)
    print(a_string)
    print(b_string)
    print(c_string)
    print(d_string)

  if method == "ReCoNet":
    cf_string = param_var(setup, 2, gammas_f, out_path)
    co_string = param_var(setup, 3, gammas_l, out_path)
    d_string = param_var(setup, 4, deltas, out_path)
    print(a_string)
    print(b_string)
    print(cf_string)
    print(co_string)
    print(d_string)
  

  #4.4.2 Epoch influence
  epoch_cst = []
  for i in range(10):
    epoch_cst.append(infer_test(setup,
               sid=2, n_styles=3, epochs=10, n_epochs=i, 
               dset='FC2', sintel_id='temple_2')[2:])
  
  epoch_cst = np.array(epoch_cst)
  
  print('st %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f' % tuple(epoch_cst[:,0]))
  print('lt %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f' % tuple(epoch_cst[:,1]))

  #4.4.3 Dataset and optical flow influence
  out_path = "C:/Users/Tom/Documents/Texmaker/MastersProject/figures/style_comp2/" + method.lower() + "/"
  cst_datasets = []
  cst_datasets.append(infer_test(setup, sid=2, n_styles=1, dset='FC2', sintel_id='temple_2', out_img_path=out_path)[2:])
  cst_datasets.append(infer_test(setup, sid=2, n_styles=1, dset='HW2', sintel_id='temple_2', out_img_path=out_path)[2:])
  cst_datasets.append(infer_test(setup, sid=2, n_styles=1, dset='CO2', sintel_id='temple_2', out_img_path=out_path)[2:])
  
  cst_datasets = np.array(cst_datasets)
  cst_datasets = np.hstack((cst_datasets[:,0], cst_datasets[:,1]))
  latex_string = " & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f" % tuple(cst_datasets)
  
  print(method + latex_string)


  #4.4.5 Multi style influence
  cst_vals = []

  cst_vals.append(infer_test(setup, sid=0, n_styles=3, sintel_id='temple_2')[2:])
  cst_vals.append(infer_test(setup, sid=1, n_styles=3, sintel_id='temple_2')[2:])
  cst_vals.append(infer_test(setup, sid=2, n_styles=3, sintel_id='temple_2')[2:])
  cst_vals.append(infer_test(setup, sid=2, n_styles=3, sintel_id='market_6')[2:])
  cst_vals.append(infer_test(setup, sid=2, n_styles=3, sintel_id='alley_2')[2:])
  
  cst_vals = np.array(cst_vals)
  cst_vals = np.hstack((cst_vals[:,0], cst_vals[:,1]))
  
  latex_string = " & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f" % tuple(cst_vals)
  print(method + latex_string)

  out_path = 'C:/Users/Tom/Documents/Texmaker/MastersProject/figures/multi-style-comp/' + method + '/'
  infer_test(setup, sid=0, n_styles=3, sintel_id='temple_2', out_img_path=out_path)
  infer_test(setup, sid=1, n_styles=3, sintel_id='temple_2', out_img_path=out_path)
  infer_test(setup, sid=2, n_styles=3, sintel_id='temple_2', out_img_path=out_path)
  
  #4.3.2 Long Term Consistency between methods
  cst_vals = []

  cst_vals.append(infer_test(setup, sid=0, sintel_id='temple_2')[3])
  cst_vals.append(infer_test(setup, sid=1, sintel_id='temple_2')[3])
  cst_vals.append(infer_test(setup, sid=2, sintel_id='temple_2')[3])
  
  cst_vals.append(infer_test(setup, sid=2, sintel_id='alley_2')[3])
  cst_vals.append(infer_test(setup, sid=2, sintel_id='market_6')[3])
  cst_vals.append(infer_test(setup, sid=2, sintel_id='temple_2')[3])
  
  latex_string = " & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f" % tuple(cst_vals)
  print(method + latex_string)
  '''

if __name__ == '__main__':
  main()