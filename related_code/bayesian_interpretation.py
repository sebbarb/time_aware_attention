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
  stat, dp, cp, dp_t, cp_t, label = get_data(data, 'ALL')

  # Get dictionaries
  static_vars, dict_dp, dict_cp = get_dictionaries(data)
  num_static = num_static(data)
  num_dp_codes, num_cp_codes = vocab_sizes(data)

  # ICD-9 Code descriptions
  dtype = {'ICD9_CODE': 'str', 'LONG_TITLE': 'str'}
  d_icd_diagnoses = pd.read_csv(hp.mimic_dir + 'D_ICD_DIAGNOSES.csv', usecols=dtype.keys(), dtype=dtype)
  d_icd_diagnoses = d_icd_diagnoses.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()
    
  dtype = {'ICD9_CODE': 'str', 'LONG_TITLE': 'str'}
  d_icd_procedures = pd.read_csv(hp.mimic_dir + 'D_ICD_PROCEDURES.csv', usecols=dtype.keys(), dtype=dtype)
  d_icd_procedures = d_icd_procedures.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()
  
  # Initialize weights
  weights_dp = pd.DataFrame(index=range(num_dp_codes), columns=['TYPE', 'ICD9_CODE', 'DESCRIPTION', 'SCORE', 'CONF_LOWER', 'CONF_UPPER'])
  weights_cp = pd.DataFrame(index=range(num_cp_codes), columns=['TYPE', 'NAME',                     'SCORE', 'CONF_LOWER', 'CONF_UPPER'])

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

  # Intervals
  all_vars = static_vars.tolist() + ['DP SCORE', 'CP SCORE']
  all_vars = [var.title() for var in all_vars]
  mu = net.fc_all.weight.mu.detach().cpu().numpy().squeeze()
  
  # For clarity, set coefficients of diag/proc/pres/charts scores to positive (and invert signs of scores accordingly)
  dp_neg, cp_neg = mu[num_static:(num_static+2)] < 0
  mu[num_static:] = np.abs(mu[num_static:])
  conf_lower = (net.fc_all.weight.mu - 1.96*net.fc_all.weight.sigma).detach().cpu().numpy().squeeze()
  conf_upper = (net.fc_all.weight.mu + 1.96*net.fc_all.weight.sigma).detach().cpu().numpy().squeeze()
  OR = np.exp(mu)
  OR_lower = np.exp(conf_lower)
  OR_upper = np.exp(conf_upper)
  
  scores_init = pd.DataFrame({'VAR': all_vars, 'COEFF': mu, 'CONF_LOWER': conf_lower, 'CONF_UPPER': conf_upper,
                              'OR': OR, 'OR_lower': OR_lower, 'OR_upper': OR_upper})

  print('Get significance of individual codes...')
  num_samples = 1000

  def pad(x): return F.pad(x, (0,1))
  bp()
  mean_coeff_dp = torch.matmul(pad(net.embed_dp.weight.mu), net.fc_dp.weight.mu.t()).detach().cpu().numpy().squeeze()
  mean_coeff_cp = torch.matmul(pad(net.embed_cp.weight.mu), net.fc_cp.weight.mu.t()).detach().cpu().numpy().squeeze()
  
  samples_coeff_dp = np.zeros((num_dp_codes, num_samples))
  samples_coeff_cp = np.zeros((num_cp_codes, num_samples))
  for sample in tqdm(range(num_samples)):
    samples_coeff_dp[:, sample] = torch.matmul(pad(net.embed_dp.weight.sample()), net.fc_dp.weight.sample().t()).detach().cpu().numpy().squeeze()
    samples_coeff_cp[:, sample] = torch.matmul(pad(net.embed_cp.weight.sample()), net.fc_cp.weight.sample().t()).detach().cpu().numpy().squeeze()

  if dp_neg:
    mean_coeff_dp *= -1 
    samples_coeff_dp *= -1 
  if cp_neg:
    mean_coeff_cp *= -1 
    samples_coeff_cp *= -1 

  ### check the following
  
  print('Diagnoses / Procedures...')
  for id_dp in tqdm(range(1, num_dp_codes)):
    name = dict_dp.get(id_dp)
    weights_dp.at[id_dp, 'TYPE']        = name[:5]
    weights_dp.at[id_dp, 'ICD9_CODE']   = name[6:]
    if name[:5] == 'DIAGN':
      weights_dp.at[id_dp, 'DESCRIPTION'] = d_icd_diagnoses.get(name[6:])
    else:
      weights_dp.at[id_dp, 'DESCRIPTION'] = d_icd_procedures.get(name[6:])
    weights_dp.at[id_dp, 'SCORE']       = mean_coeff_dp[id_dp]
    weights_dp.at[id_dp, 'CONF_LOWER']  = np.percentile(samples_coeff_dp[id_dp], 5)
    weights_dp.at[id_dp, 'CONF_UPPER']  = np.percentile(samples_coeff_dp[id_dp], 95)
    
  print('Charts / Prescriptions...')
  for id_cp in tqdm(range(1, num_cp_codes)):
    name = dict_cp.get(id_cp)
    weights_cp.at[id_cp, 'TYPE']        = name[:5]
    weights_cp.at[id_cp, 'NAME']        = name[6:]
    weights_cp.at[id_cp, 'SCORE']       = mean_coeff_cp[id_cp]
    weights_cp.at[id_cp, 'CONF_LOWER']  = np.percentile(samples_coeff_cp[id_cp], 5)
    weights_cp.at[id_cp, 'CONF_UPPER']  = np.percentile(samples_coeff_cp[id_cp], 95)

  print('-----------------------------------------')
  print(scores_init)
  print('-----------------------------------------')
  weights_dp.sort_values(by='SCORE', ascending=False, inplace=True)
  print(weights_dp.head(10))
  print('-----------------------------------------')
  weights_cp.sort_values(by='SCORE', ascending=False, inplace=True)
  print(weights_cp.head(10))
  print('-----------------------------------------')
  
  print('Save...')

  scores_init.to_pickle(hp.data_dir + 'scores_init_test.pkl')
  scores_init.to_csv(hp.data_dir + 'scores_init_test.csv', index=False)
  
  weights_dp.to_pickle(hp.data_dir + 'weights_dp.pkl')
  weights_dp.to_csv(hp.data_dir + 'weights_dp.csv', index=False)

  weights_cp.to_pickle(hp.data_dir + 'weights_cp.pkl')
  weights_cp.to_csv(hp.data_dir + 'weights_cp.csv', index=False)

