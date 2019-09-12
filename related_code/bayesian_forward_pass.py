'''
Mar 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
import numpy as np
import pandas as pd
from hyperparameters import Hyperparameters as hp
from data_load import *
from modules import *
from bayesian_train import *
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pdb import set_trace as bp

if __name__ == '__main__':
  # Load data
  print('Load data...')
  data = np.load(hp.data_dir + 'data_arrays.npz')
  stat, diag, proc, pres, charts, diag_t, proc_t, label = get_data(data, 'ALL')

  num_static = stat.shape[1]
  num_diag_codes, num_proc_codes, num_pres_codes = vocab_sizes(data)
  num_charts = charts.shape[1]
  
  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda:0' if use_cuda else 'cpu')
  hp.device = device
  torch.backends.cudnn.benchmark = True
  
  # Set log dir to read trained model from
  logdir = hp.logdir + 'bayesian_all/'

  # Create dataset for dataloader
  dataset = utils.TensorDataset(torch.from_numpy(stat), 
                                torch.from_numpy(diag),
                                torch.from_numpy(proc),
                                torch.from_numpy(pres),
                                torch.from_numpy(charts),
                                torch.from_numpy(diag_t),
                                torch.from_numpy(proc_t),
                                torch.from_numpy(label))

  # Compute total batch count
  num_batches = len(label) // hp.batch_size

  # Create batch queues, don't shuffle
  trainloader = utils.DataLoader(dataset,
                                 batch_size = hp.batch_size, 
                                 shuffle = None,
                                 sampler = None,
                                 num_workers = 2,
                                 drop_last = False)

  # Weight of positive samples for training
  pos_weight = torch.tensor((len(label) - np.sum(label))/np.sum(label))

  # Pytorch Network
  net = BayesianNetwork(num_static, num_diag_codes, num_proc_codes, num_pres_codes, num_charts, pos_weight, num_batches, output_activations=True).to(device)

  # Restore variables from disk
  net.load_state_dict(torch.load(logdir + 'final_model.pt', map_location=torch.device('cpu')))

  # # Get output of one forward pass
  # print('Get parameters...')  
  # net.eval()
  # for i, (stat, diag, proc, pres, charts, diag_t, proc_t, label) in enumerate(tqdm(trainloader), 0):
    # with torch.no_grad():
      # label_pred, other = net(stat.to(device), diag.to(device), proc.to(device), pres.to(device), charts.to(device), diag_t.to(device), proc_t.to(device))
    # label_pred = label_pred.unsqueeze(-1).cpu().numpy()
    # other = [o.cpu().numpy() for o in other]
    # weights_diag, weights_proc, weights_pres, charts_activations, all = other
    # if (i == 0):
      # label_pred_stack = label_pred
      # weights_diag_stack = weights_diag
      # weights_proc_stack = weights_proc
      # weights_pres_stack = weights_pres
      # charts_activations_stack = charts_activations
      # all_stack = all
    # else:
      # label_pred_stack = np.vstack((label_pred_stack, label_pred))
      # weights_diag_stack = np.vstack((weights_diag_stack, weights_diag))
      # weights_proc_stack = np.vstack((weights_proc_stack, weights_proc))
      # weights_pres_stack = np.vstack((weights_pres_stack, weights_pres))
      # charts_activations_stack = np.vstack((charts_activations_stack, charts_activations))
      # all_stack = np.vstack((all_stack, all))
  
  # print('Save...')
  # np.savez(hp.data_dir + 'data_fp.npz', label_pred=label_pred_stack, weights_diag=weights_diag_stack, weights_proc=weights_proc_stack, weights_pres=weights_pres_stack, 
           # charts_activations=charts_activations_stack, all=all_stack)
  label_pred = np.load(hp.data_dir + 'data_fp.npz')['label_pred']

  # Platt scaling
  clf = LogisticRegression().fit(label_pred, label)
  label_sigmoids = clf.predict_proba(label_pred)[:, 1]
  
  # # Calibration
  # plt.plot([0, 1], [0, 1], 'k:')
  # fraction_of_positives, mean_predicted_value = calibration_curve(label, label_sigmoids, n_bins=10)
  # plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Calibrated')
  # fraction_of_positives, mean_predicted_value = calibration_curve(label, expit(label_pred), n_bins=10)
  # plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Original')
  # ax = plt.gca()
  # ax.set_xlabel('Mean Predicted Probability')
  # ax.set_ylabel('True Probability (Fractions of Positives)')
  # ax.set_ylim([-0.05, 1.05])
  # ax.legend(loc='upper left')
  # ax.set_title('Calibration plots (reliability curve)')
  # plt.show()  

  # Pytorch Network
  net = BayesianNetwork(num_static, num_diag_codes, num_proc_codes, num_pres_codes, num_charts, pos_weight, num_batches).to(device)

  # Restore variables from disk
  net.load_state_dict(torch.load(logdir + 'final_model.pt', map_location=torch.device('cpu')))

  # Get output of one forward pass
  num_samples = 100
  print('Get parameters...')  
  net.eval()
  for i, (stat, diag, proc, pres, charts, diag_t, proc_t, label) in enumerate(tqdm(trainloader), 0):
    # move to GPU if available
    stat   = stat.to(device)
    diag   = diag.to(device)
    proc   = proc.to(device)
    pres   = pres.to(device)
    charts = charts.to(device)
    diag_t = diag_t.to(device)
    proc_t = proc_t.to(device)
    label  = label.to(device)
      
    pred_samples = np.zeros((len(label), num_samples))
    for sample in range(num_samples):
      with torch.no_grad():
        pred_sample, _ = net(stat, diag, proc, pres, charts, diag_t, proc_t, sample=True)
      pred_samples[:, sample] = clf.predict_proba(pred_sample.unsqueeze(-1).detach().cpu().numpy())[:, 1]
    if (i == 0):
      pred_samples_stack = pred_samples
    else:
      pred_samples_stack = np.vstack((pred_samples_stack, pred_samples))

  np.savez(hp.data_dir + 'data_fp_samples.npz', label_sigmoids=label_sigmoids, pred_samples=pred_samples_stack)
  