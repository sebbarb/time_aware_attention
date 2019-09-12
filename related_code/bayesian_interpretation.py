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
  data = np.load(hp.data_dir + 'data_arrays.npz')
  trainloader, num_batches, pos_weight = get_trainloader(data, 'ALL')
  stat, diag, proc, pres, charts, diag_t, proc_t, label = get_data(data, 'ALL')

  # Get dictionaries
  static_vars, dict_diagnoses, dict_procedures, dict_prescriptions, charts_columns = get_dictionaries(data)
  num_static = stat.shape[1]
  num_diag_codes, num_proc_codes, num_pres_codes = vocab_sizes(data)
  num_charts = charts.shape[1]

  # ICD-9 Code descriptions
  dtype = {'ICD9_CODE': 'str', 'LONG_TITLE': 'str'}
  d_icd_diagnoses = pd.read_csv(hp.mimic_dir + 'D_ICD_DIAGNOSES.csv', usecols=dtype.keys(), dtype=dtype)
  d_icd_diagnoses = d_icd_diagnoses.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()
    
  dtype = {'ICD9_CODE': 'str', 'LONG_TITLE': 'str'}
  d_icd_procedures = pd.read_csv(hp.mimic_dir + 'D_ICD_PROCEDURES.csv', usecols=dtype.keys(), dtype=dtype)
  d_icd_procedures = d_icd_procedures.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()
  
  # Initialize weights
  weights_diag = pd.DataFrame(index=range(num_diag_codes), columns=['ICD9_CODE', 'DESCRIPTION', 'SCORE', 'CONF_LOWER', 'CONF_UPPER'])
  weights_proc = pd.DataFrame(index=range(num_proc_codes), columns=['ICD9_CODE', 'DESCRIPTION', 'SCORE', 'CONF_LOWER', 'CONF_UPPER'])
  weights_pres = pd.DataFrame(index=range(num_diag_codes), columns=['DRUG',                     'SCORE', 'CONF_LOWER', 'CONF_UPPER'])

  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda:0' if use_cuda else 'cpu')
  hp.device = device
  torch.backends.cudnn.benchmark = True

  # Pytorch Network
  net = BayesianNetwork(num_static, num_diag_codes, num_proc_codes, num_pres_codes, num_charts, pos_weight, num_batches).to(device)
  net.eval()
  
  # Set log dir to read trained model from
  logdir = hp.logdir + 'bayesian_all/'
  
  # Restore variables from disk
  net.load_state_dict(torch.load(logdir + 'final_model.pt', map_location=torch.device(device)))

  # Intervals
  all_vars = static_vars.tolist() + ['DIAGNOSES SCORE', 'PROCEDURES SCORE', 'PRESCRIPTIONS SCORE'] + charts_columns.tolist()
  all_vars = [var.title() for var in all_vars]
  mu = net.fc_all.weight.mu.detach().cpu().numpy().squeeze()
  # For clarity, set coefficients of diag/proc/pres/charts scores to positive (and invert signs of scores accordingly)
  diag_neg, proc_neg, pres_neg = mu[num_static:(num_static+3)] < 0
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
  
  mean_coeff_diag = torch.matmul(net.embed_diag.weight.mu, net.fc_diag.weight.mu.t()).detach().cpu().numpy().squeeze()
  mean_coeff_proc = torch.matmul(net.embed_proc.weight.mu, net.fc_proc.weight.mu.t()).detach().cpu().numpy().squeeze()
  mean_coeff_pres = torch.matmul(net.embed_pres.weight.mu, net.fc_pres.weight.mu.t()).detach().cpu().numpy().squeeze()  
  
  samples_coeff_diag = np.zeros((num_diag_codes, num_samples))
  samples_coeff_proc = np.zeros((num_proc_codes, num_samples))
  samples_coeff_pres = np.zeros((num_pres_codes, num_samples))
  for sample in tqdm(range(num_samples)):
    samples_coeff_diag[:, sample] = torch.matmul(net.embed_diag.weight.sample(), net.fc_diag.weight.sample().t()).detach().cpu().numpy().squeeze()
    samples_coeff_proc[:, sample] = torch.matmul(net.embed_proc.weight.sample(), net.fc_proc.weight.sample().t()).detach().cpu().numpy().squeeze()
    samples_coeff_pres[:, sample] = torch.matmul(net.embed_pres.weight.sample(), net.fc_pres.weight.sample().t()).detach().cpu().numpy().squeeze()
    
  if diag_neg:
    mean_coeff_diag *= -1 
    samples_coeff_diag *= -1 
  if proc_neg:
    mean_coeff_proc *= -1 
    samples_coeff_proc *= -1 
  if pres_neg:
    mean_coeff_pres *= -1 
    samples_coeff_pres *= -1 
  
  print('Diagnoses...')
  for id_diag in tqdm(range(1, num_diag_codes)):
    weights_diag.at[id_diag, 'ICD9_CODE']   = dict_diagnoses.get(id_diag)
    weights_diag.at[id_diag, 'DESCRIPTION'] = d_icd_diagnoses.get(dict_diagnoses.get(id_diag))
    weights_diag.at[id_diag, 'SCORE']       = mean_coeff_diag[id_diag]
    weights_diag.at[id_diag, 'CONF_LOWER']  = np.percentile(samples_coeff_diag[id_diag], 5)
    weights_diag.at[id_diag, 'CONF_UPPER']  = np.percentile(samples_coeff_diag[id_diag], 95)
  
  print('Procedures...')
  for id_proc in tqdm(range(1, num_proc_codes)):
    weights_proc.at[id_proc, 'ICD9_CODE']   = dict_procedures.get(id_proc)
    weights_proc.at[id_proc, 'DESCRIPTION'] = d_icd_procedures.get(dict_procedures.get(id_proc))
    weights_proc.at[id_proc, 'SCORE']       = mean_coeff_proc[id_proc]
    weights_proc.at[id_proc, 'CONF_LOWER']  = np.percentile(samples_coeff_proc[id_proc], 5)
    weights_proc.at[id_proc, 'CONF_UPPER']  = np.percentile(samples_coeff_proc[id_proc], 95)
    
  print('Prescriptions...')
  for id_pres in tqdm(range(1, num_pres_codes)):
    weights_pres.at[id_pres, 'DRUG']        = dict_prescriptions.get(id_pres)
    weights_pres.at[id_pres, 'SCORE']       = mean_coeff_pres[id_pres]
    weights_pres.at[id_pres, 'CONF_LOWER']  = np.percentile(samples_coeff_pres[id_pres], 5)
    weights_pres.at[id_pres, 'CONF_UPPER']  = np.percentile(samples_coeff_pres[id_pres], 95)

  print('-----------------------------------------')
  print(scores_init)
  print('-----------------------------------------')
  weights_diag.sort_values(by='SCORE', ascending=False, inplace=True)
  print(weights_diag.head(10))
  print('-----------------------------------------')
  weights_proc.sort_values(by='SCORE', ascending=False, inplace=True)
  print(weights_proc.head(10))
  print('-----------------------------------------')
  weights_pres.sort_values(by='SCORE', ascending=False, inplace=True)
  print(weights_pres.head(10))
  print('-----------------------------------------')
  
  print('Save...')
  scores_init.to_pickle(hp.data_dir + 'scores_init_test.pkl')
  scores_init.to_csv(hp.data_dir + 'scores_init_test.csv', index=False)
  
  weights_diag.to_pickle(hp.data_dir + 'weights_diag.pkl')
  weights_diag.to_csv(hp.data_dir + 'weights_diag.csv', index=False)
  
  weights_proc.to_pickle(hp.data_dir + 'weights_proc.pkl')
  weights_proc.to_csv(hp.data_dir + 'weights_proc.csv', index=False)

  weights_pres.to_pickle(hp.data_dir + 'weights_pres.pkl')
  weights_pres.to_csv(hp.data_dir + 'weights_pres.csv', index=False)

