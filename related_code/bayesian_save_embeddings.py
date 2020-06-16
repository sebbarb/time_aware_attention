'''
Mar 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
import numpy as np
import pandas as pd
from hyperparameters import Hyperparameters as hp
from data_load import *
from bayesian_train import *
from tqdm import tqdm
import statsmodels.api as sm
from pdb import set_trace as bp

if __name__ == '__main__':
  # Load data
  print('Load data...')
  data = np.load(hp.data_dir + 'data_arrays.npz', allow_pickle=True)
  trainloader, num_batches, pos_weight = get_trainloader(data, 'ALL')

  # Get dictionaries
  static_vars, dict_dp, dict_cp = get_dictionaries(data)
  num_static = num_static(data)
  num_dp_codes, num_cp_codes = vocab_sizes(data)
  
  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda:0' if use_cuda else 'cpu')
  hp.device = device
  torch.backends.cudnn.benchmark = True

  # Pytorch Network
  net = BayesianNetwork(num_static, num_dp_codes, num_cp_codes, pos_weight, num_batches).to(device)
  net.eval()
  
  # Set log dir to read trained model from
  logdir = hp.logdir + 'bayesian_all/'
  
  # Restore variables from disk
  net.load_state_dict(torch.load(logdir + 'final_model.pt', map_location=torch.device(device)))

  # Embeddings
  embed_dp = net.embed_dp.weight.mu.detach().cpu().numpy().squeeze()
  embed_cp = net.embed_cp.weight.mu.detach().cpu().numpy().squeeze()

  # Data frames
  df_embed_dp = pd.DataFrame(data = embed_dp)
  df_embed_cp = pd.DataFrame(data = embed_cp)

  ### check the following
  bp()
  print('Diagnoses / Procedures...')
  for id_dp in tqdm(range(1, num_dp_codes)):
    name = dict_dp.get(id_dp)
    df_embed_dp.at[id_dp, 'TYPE']        = name[:5]
    df_embed_dp.at[id_dp, 'ICD9_CODE']   = name[6:]
    
  print('Charts / Prescriptions...')
  for id_cp in tqdm(range(1, num_cp_codes)):
    name = dict_cp.get(id_cp)
    df_embed_cp.at[id_cp, 'TYPE']        = name[:5]
    df_embed_cp.at[id_cp, 'NAME']        = name[6:]

  print('Save...')

  df_embed_dp.to_pickle(hp.data_dir + 'df_embed_dp.pkl')
  df_embed_dp.to_csv(hp.data_dir + 'df_embed_dp.csv', index=False)

  df_embed_cp.to_pickle(hp.data_dir + 'df_embed_cp.pkl')
  df_embed_cp.to_csv(hp.data_dir + 'df_embed_cp.csv', index=False)

