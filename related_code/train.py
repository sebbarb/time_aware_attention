'''
Mar 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from hyperparameters import Hyperparameters as hp
from data_load import *
from modules import *
import os
from tqdm import tqdm
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, roc_auc_score, f1_score
from pdb import set_trace as bp


if __name__ == '__main__':
  # Load data
  print('Load data...')
  data = np.load(hp.data_dir + 'data_arrays.npz', allow_pickle=True)
  
  # Training and validation data
  if hp.all_train:
    trainloader, num_batches, pos_weight = get_trainloader(data, 'ALL')
  else:
    trainloader, num_batches, pos_weight = get_trainloader(data, 'TRAIN')
  
  # Vocabulary sizes
  num_static = num_static(data)
  num_dp_codes, num_cp_codes = vocab_sizes(data)
  
  print('-----------------------------------------')
  print('Train...')

  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda:0' if use_cuda else 'cpu')
  torch.backends.cudnn.benchmark = True

  # Network
  net = Net(num_static, num_dp_codes, num_cp_codes).to(device)

  # Loss function and optimizer
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
  optimizer = optim.Adam(net.parameters(), lr = 0.001)  

  # Create log dir
  logdir = hp.logdir + hp.net_variant + '/'
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  
  # Store times
  epoch_times = []

  # Train
  for epoch in tqdm(range(hp.num_epochs)): 
    # print('-----------------------------------------')
    # print('Epoch: {}'.format(epoch))
    net.train()
    time_start = time()
    for i, (stat, dp, cp, dp_t, cp_t, label) in enumerate(tqdm(trainloader), 0):
      # move to GPU if available
      stat  = stat.to(device)
      dp    = dp.to(device)
      cp    = cp.to(device)
      dp_t  = dp_t.to(device)
      cp_t  = cp_t.to(device)
      label = label.to(device)
    
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      label_pred, _ = net(stat, dp, cp, dp_t, cp_t)
      loss = criterion(label_pred, label)
      loss.backward()
      optimizer.step()
    
    # timing
    time_end = time()
    epoch_times.append(time_end-time_start)

  # Save
  print('Saving...')
  torch.save(net.state_dict(), logdir + 'final_model.pt')
  np.savez(logdir + 'epoch_times', epoch_times=epoch_times)
  print('Done')
  
