# -*- coding: utf-8 -*-

from fs_tests import train_net, infer_test, select_method, eval_test

import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='models/raft-chairs.pth', help="restore checkpoint")
  parser.add_argument('--path', default='demo-frames', help="dataset for evaluation")
  parser.add_argument('--small', action='store_true', help='use small model')
  parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
  parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
  args = parser.parse_args()
  
  #method = "Johnson"
  #method = "Huang"
  method = "Dumoulin"
  #method = "Ruder"
  setup = select_method(method)
  
  #train_net(setup, sid=2, epochs=3)
  #infer_test(setup, sid=2, n_styles=1)
  '''
  def eval_test(setup, args,
              n_styles=1, epochs=3, n_epochs=2, 
              batchsize=16, learning_rate=1e-3):
  '''
  eval_test(setup, args, n_styles=3, epochs=20, n_epochs=19)
  #eval_test(setup, args, n_styles=1, epochs=20, n_epochs=19)
  
if __name__ == '__main__':
  main()