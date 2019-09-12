'''
Mar 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
import numpy as np
import pandas as pd
from hyperparameters import Hyperparameters as hp
from data_load import *
from modules import *
from tqdm import tqdm
import statsmodels.api as sm
from pdb import set_trace as bp

if __name__ == '__main__':
  # Load data
  print('Load data...')
  data = np.load(hp.data_dir + 'data_arrays.npz')
  stat, diag, proc, pres, charts, diag_t, proc_t, label = get_data(data, 'ALL')
  # stat, diag, proc, pres, charts, diag_t, proc_t, label = get_data(data, 'TEST')

  # Pytorch Network
  num_static = stat.shape[1]
  num_diag_codes, num_proc_codes, num_pres_codes = vocab_sizes(data)
  num_charts = charts.shape[1]
  net = Net(num_static, num_diag_codes, num_proc_codes, num_pres_codes, num_charts, output_activations=True)
  
  # Set log dir to read trained model from
  logdir = hp.logdir + 'attention_all/0/'
  # logdir = hp.logdir + 'attention/0/'
  
  # Restore variables from disk
  net.load_state_dict(torch.load(logdir + 'final_model.pt', map_location=torch.device('cpu')))

  # Create dataset for dataloader
  dataset = utils.TensorDataset(torch.from_numpy(stat), 
                                torch.from_numpy(diag),
                                torch.from_numpy(proc),
                                torch.from_numpy(pres),
                                torch.from_numpy(charts),
                                torch.from_numpy(diag_t),
                                torch.from_numpy(proc_t),
                                torch.from_numpy(label))

  # Create batch queues, don't shuffle
  trainloader = utils.DataLoader(dataset,
                                 batch_size = hp.batch_size, 
                                 shuffle = None,
                                 sampler = None,
                                 num_workers = 2,
                                 drop_last = False)

  # Get output of one forward pass
  print('Get parameters...')  
  net.eval()
  for i, (stat, diag, proc, pres, charts, diag_t, proc_t, label) in enumerate(tqdm(trainloader), 0):
    with torch.no_grad():
      label_pred, other = net(stat, diag, proc, pres, charts, diag_t, proc_t)
    other = [o.numpy() for o in other]
    weights_diag, weights_proc, weights_pres, score_diag, score_proc, score_pres, charts_activations, all = other
    if (i == 0):
      weights_diag_stack = weights_diag
      weights_proc_stack = weights_proc
      weights_pres_stack = weights_pres
      score_diag_stack = score_diag
      score_proc_stack = score_proc
      score_pres_stack = score_pres
      charts_activations_stack = charts_activations
      all_stack = all
    else:
      weights_diag_stack = np.vstack((weights_diag_stack, weights_diag))
      weights_proc_stack = np.vstack((weights_proc_stack, weights_proc))
      weights_pres_stack = np.vstack((weights_pres_stack, weights_pres))
      score_diag_stack = np.vstack((score_diag_stack, score_diag))
      score_proc_stack = np.vstack((score_proc_stack, score_proc))
      score_pres_stack = np.vstack((score_pres_stack, score_pres))
      charts_activations_stack = np.vstack((charts_activations_stack, charts_activations))
      all_stack = np.vstack((all_stack, all))
  
  print('Save...')
  np.savez(hp.data_dir + 'data_fp.npz', weights_diag=weights_diag_stack, weights_proc=weights_proc_stack, weights_pres=weights_pres_stack, 
           score_diag=score_diag_stack, score_proc=score_proc_stack, score_pres=score_pres_stack, charts_activations=charts_activations_stack, all=all_stack)
  # np.savez(hp.data_dir + 'data_fp_test.npz', weights_diag=weights_diag_stack, weights_proc=weights_proc_stack, weights_pres=weights_pres_stack, 
           # score_diag=score_diag_stack, score_proc=score_proc_stack, score_pres=score_pres_stack, charts_activations=charts_activations_stack, all=all_stack)
           
           