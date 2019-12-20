'''
Aug 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
from __future__ import print_function
from hyperparameters import Hyperparameters as hp
import numpy as np
import torch
import torch.utils.data as utils
from pdb import set_trace as bp


def get_data(data, type):
  # Data
  static       = data['static'].astype('float32')
  label        = data['label'].astype('float32')
  dp           = data['dp'].astype('int64') # diagnoses/procedures
  cp           = data['cp'].astype('int64') # charts/prescriptions
  dp_times     = data['dp_times'].astype('float32')
  cp_times     = data['cp_times'].astype('float32')
  train_ids    = data['train_ids']
  validate_ids = data['validate_ids']
  test_ids     = data['test_ids']  

  if (type == 'TRAIN'):
    ids = train_ids
  elif (type == 'VALIDATE'):
    ids = validate_ids
  elif (type == 'TEST'):
    ids = test_ids
  elif (type == 'ALL'):
    ids = np.full_like(label, True, dtype=bool)

  static   = static[ids, :]
  label    = label[ids]
  dp       = dp[ids, :]
  cp       = cp[ids, :]
  dp_times = dp_times[ids, :]
  cp_times = cp_times[ids, :]
  
  return static, dp, cp, dp_times, cp_times, label


def get_dictionaries(data):
  return data['static_vars'], data['dict_dp'][()], data['dict_cp'][()]


def num_static(data):
  return data['static_vars'].shape[0]
  

def vocab_sizes(data):
  return data['dp'].max()+1, data['cp'].max()+1

  
def get_trainloader(data, type, shuffle=True, idx=None):
  # Data
  static, dp, cp, dp_times, cp_times, label = get_data(data, type)

  # Bootstrap
  if idx is not None:
    static, dp, cp, dp_times, cp_times, label = static[idx], dp[idx], cp[idx], dp_times[idx], cp_times[idx], label[idx]

  # Compute total batch count
  num_batches = len(label) // hp.batch_size
  
  # Create dataset
  dataset = utils.TensorDataset(torch.from_numpy(static), 
                                torch.from_numpy(dp),
                                torch.from_numpy(cp),
                                torch.from_numpy(dp_times),
                                torch.from_numpy(cp_times),
                                torch.from_numpy(label))

  # Create batch queues
  trainloader = utils.DataLoader(dataset,
                                 batch_size = hp.batch_size, 
                                 shuffle = shuffle,
                                 sampler = None,
                                 num_workers = 2,
                                 drop_last = True)
                                 
  # Weight of positive samples for training
  pos_weight = torch.tensor((len(label) - np.sum(label))/np.sum(label))
  
  return trainloader, num_batches, pos_weight
  
  
if __name__ == '__main__':
  print('Load data...')
  data = np.load(hp.data_dir + 'data_arrays.npz')
  trainloader, num_batches, pos_weight = get_trainloader(data, 'TRAIN')
  # vocab_diagnoses, vocab_procedures, vocab_prescriptions = vocab_sizes(data)
  
  