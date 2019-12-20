'''
Feb 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
import torch
import math

class Hyperparameters:
  '''Hyperparameters'''
  # data
  mimic_dir = '../../MIMIC-III Clinical Database/uncompressed/'
  data_dir = '../data/'
  logdir = '../logdir/' # log directory  

  # model
  min_count = 100 # words whose occurred less than min_cnt are encoded as OTHER
    
  # training
  batch_size = 128
  num_epochs = 80
  dropout_rate = 0.5
  patience = 10 # early stopping
  
  # which data to load
  # on_the_cloud = False
  all_train = False
  # all_train = True
  
  # network variants
  # net_variant = 'birnn_concat_time_delta'
  # net_variant = 'birnn_concat_time_delta_attention'
  # net_variant = 'birnn_time_decay'#
  # net_variant = 'birnn_time_decay_attention'#
  # net_variant = 'ode_birnn'#
  # net_variant = 'ode_birnn_attention'#
  # net_variant = 'ode_attention'#
  # net_variant = 'attention_concat_time'
  net_variant = 'birnn_ode_decay'#
  # net_variant = 'birnn_ode_decay_attention'#
  # net_variant = 'mce_attention'#
  # net_variant = 'mce_birnn'#
  # net_variant = 'mce_birnn_attention'#
  
  # bootstrapping
  np_seed = 1234
  bootstrap_samples = 100
  
  # bayesian network
  pi = 0.5
  sigma1 = math.exp(-0)
  sigma2 = math.exp(-6)
  samples = 1
  test_samples = 10
  
  
  
  

  
  