'''
Mar 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
from __future__ import print_function
import torch
import numpy as np
import pandas as pd
import pickle
import scipy.stats as st
from hyperparameters import Hyperparameters as hp
from data_load import *
from modules import *
import os
from tqdm import tqdm
from train import Net
#import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.calibration import calibration_curve
from pdb import set_trace as bp

def round(num):
  return np.round(num*1000)/1000

if __name__ == '__main__':
  # Load data
  print('Load data...')
  data = np.load(hp.data_dir + 'data_arrays.npz')
  test_ids_patients = pd.read_pickle(hp.data_dir + 'test_ids_patients.pkl')
  
  # Patients in test data
  patients = test_ids_patients.drop_duplicates()
  num_patients = patients.shape[0]
  row_ids = pd.DataFrame({'ROW_IDX': test_ids_patients.index}, index=test_ids_patients)
  
  # Vocabulary sizes
  num_static = num_static(data)
  num_dp_codes, num_cp_codes = vocab_sizes(data)

  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda:0' if use_cuda else 'cpu')
  torch.backends.cudnn.benchmark = True

  # Network
  net = Net(num_static, num_dp_codes, num_cp_codes).to(device)
  
  print('Evaluate...')
  # Set log dir to read trained model from
  logdir = hp.logdir + hp.net_variant + '/'

  # Restore variables from disk
  net.load_state_dict(torch.load(logdir + 'final_model.pt', map_location=device))

  # Bootstrapping
  np.random.seed(hp.np_seed)
  avpre_vec = np.zeros(hp.bootstrap_samples)
  auroc_vec = np.zeros(hp.bootstrap_samples)
  f1_vec    = np.zeros(hp.bootstrap_samples)
  sensitivity_vec = np.zeros(hp.bootstrap_samples)
  specificity_vec = np.zeros(hp.bootstrap_samples)
  ppv_vec = np.zeros(hp.bootstrap_samples)
  npv_vec = np.zeros(hp.bootstrap_samples)  
  
  for sample in range(hp.bootstrap_samples):
    print('Bootstrap sample {}'.format(sample))
  
    # Test data
    sample_patients = patients.sample(n=num_patients, replace=True)
    idx = np.squeeze(row_ids.loc[sample_patients].values)
    testloader, _, _ = get_trainloader(data, 'TEST', shuffle=False, idx=idx)

    # evaluate on test data
    net.eval()
    label_pred = torch.Tensor([])
    label_test = torch.Tensor([])
    with torch.no_grad():
      for i, (stat, dp, cp, dp_t, cp_t, label_batch) in enumerate(tqdm(testloader), 0):
        # move to GPU if available
        stat  = stat.to(device)
        dp    = dp.to(device)
        cp    = cp.to(device)
        dp_t  = dp_t.to(device)
        cp_t  = cp_t.to(device)
      
        label_pred_batch, _ = net(stat, dp, cp, dp_t, cp_t)
        label_pred = torch.cat((label_pred, label_pred_batch.cpu()))
        label_test = torch.cat((label_test, label_batch))
        
    label_sigmoids = torch.sigmoid(label_pred).cpu().numpy()

    # Average precision
    avpre = average_precision_score(label_test, label_sigmoids)
    
    # Determine AUROC score
    auroc = roc_auc_score(label_test, label_sigmoids)

    # Sensitivity, specificity
    fpr, tpr, thresholds = roc_curve(label_test, label_sigmoids)
    youden_idx = np.argmax(tpr - fpr)
    sensitivity = tpr[youden_idx]
    specificity = 1-fpr[youden_idx]
    
    # F1, PPV, NPV score
    f1 = 0
    ppv = 0
    npv = 0
    for t in thresholds:
      label_pred = (np.array(label_sigmoids) >= t).astype(int)
      f1_temp = f1_score(label_test, label_pred)
      ppv_temp = precision_score(label_test, label_pred, pos_label=1)
      npv_temp = precision_score(label_test, label_pred, pos_label=0)
      if f1_temp > f1:
        f1 = f1_temp
      if (ppv_temp+npv_temp) > (ppv+npv):
        ppv = ppv_temp
        npv = npv_temp

    # Store in vectors
    avpre_vec[sample] = avpre
    auroc_vec[sample] = auroc
    f1_vec[sample]    = f1
    sensitivity_vec[sample]  = sensitivity
    specificity_vec[sample]  = specificity
    ppv_vec[sample]  = ppv
    npv_vec[sample]  = npv    

  avpre_mean = np.mean(avpre_vec)
  avpre_lci, avpre_uci = st.t.interval(0.95, hp.bootstrap_samples-1, loc=avpre_mean, scale=st.sem(avpre_vec))
  auroc_mean = np.mean(auroc_vec)
  auroc_lci, auroc_uci = st.t.interval(0.95, hp.bootstrap_samples-1, loc=auroc_mean, scale=st.sem(auroc_vec))
  f1_mean = np.mean(f1_vec)
  f1_lci, f1_uci = st.t.interval(0.95, hp.bootstrap_samples-1, loc=f1_mean, scale=st.sem(f1_vec))
  ppv_mean = np.mean(ppv_vec)
  ppv_lci, ppv_uci = st.t.interval(0.95, hp.bootstrap_samples-1, loc=ppv_mean, scale=st.sem(ppv_vec))
  npv_mean = np.mean(npv_vec)
  npv_lci, npv_uci = st.t.interval(0.95, hp.bootstrap_samples-1, loc=npv_mean, scale=st.sem(npv_vec))    
  sensitivity_mean = np.mean(sensitivity_vec)
  sensitivity_lci, sensitivity_uci = st.t.interval(0.95, hp.bootstrap_samples-1, loc=sensitivity_mean, scale=st.sem(sensitivity_vec))
  specificity_mean = np.mean(specificity_vec)
  specificity_lci, specificity_uci = st.t.interval(0.95, hp.bootstrap_samples-1, loc=specificity_mean, scale=st.sem(specificity_vec))

  epoch_times = np.load(hp.logdir + hp.net_variant + '/epoch_times.npz')['epoch_times']
  times_mean = np.mean(epoch_times)
  times_lci, times_uci = st.t.interval(0.95, len(epoch_times)-1, loc=np.mean(epoch_times), scale=st.sem(epoch_times))
  times_std = np.std(epoch_times)
  
  print('------------------------------------------------')
  print('Net variant: {}'.format(hp.net_variant))
  print('Average Precision: {} [{},{}]'.format(round(avpre_mean), round(avpre_lci), round(avpre_uci)))
  print('AUROC: {} [{},{}]'.format(round(auroc_mean), round(auroc_lci), round(auroc_uci)))
  print('F1: {} [{},{}]'.format(round(f1_mean), round(f1_lci), round(f1_uci)))  
  print('PPV: {} [{},{}]'.format(round(ppv_mean), round(ppv_lci), round(ppv_uci)))
  print('NPV: {} [{},{}]'.format(round(npv_mean), round(npv_lci), round(npv_uci)))  
  print('Sensitivity: {} [{},{}]'.format(round(sensitivity_mean), round(sensitivity_lci), round(sensitivity_uci)))
  print('Specificity: {} [{},{}]'.format(round(specificity_mean), round(specificity_lci), round(specificity_uci)))
  print('Time: {} [{},{}] std: {}'.format(round(times_mean), round(times_lci), round(times_uci), round(times_std)))
  print('Done')
  

