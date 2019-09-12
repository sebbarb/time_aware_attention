'''
Mar 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
from __future__ import print_function
import pandas as pd
import numpy as np
import scipy.stats as st
from hyperparameters import Hyperparameters as hp
from data_load import *
from modules import *
import os
from tqdm import tqdm
from train import Net
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.calibration import calibration_curve
from sklearn import linear_model
from pdb import set_trace as bp

def round(num):
  return np.round(num*1000)/1000

if __name__ == '__main__':
  # Load icu_pat table
  print('Loading icu_pat...')
  icu_pat = pd.read_pickle(hp.data_dir + 'icu_pat_admit.pkl')
  
  print('Loading last vital signs measurements...')
  charts = pd.read_pickle(hp.data_dir + 'charts_outputs_last_only.pkl')
  charts = charts.drop(columns=['CHARTTIME'])
  charts = pd.get_dummies(charts, columns = ['VALUECAT']).groupby('ICUSTAY_ID').sum()
  charts.drop(columns=['VALUECAT_BM_n', 'VALUECAT_BP_n', 'VALUECAT_BT_n', 'VALUECAT_GC_n', 'VALUECAT_HR_n', 'VALUECAT_RR_n', 'VALUECAT_UO_n'], inplace=True) # drop reference columns
  
  print('-----------------------------------------')
  
  print('Create array of static variables...')
  
  num_icu_stays = len(icu_pat['ICUSTAY_ID'])
  
  # static variables
  print('Create static array...')
  icu_pat = pd.get_dummies(icu_pat, columns = ['ADMISSION_LOCATION', 'INSURANCE', 'MARITAL_STATUS', 'ETHNICITY'])
  icu_pat.drop(columns=['ADMISSION_LOCATION_Emergency Room Admit', 'INSURANCE_Medicare', 'MARITAL_STATUS_Married/Life Partner', 'ETHNICITY_White'], inplace=True) # drop reference columns
  
  # merge with last vital signs measurements
  icu_pat = pd.merge(icu_pat, charts, how='left', on='ICUSTAY_ID').fillna(0)
  
  static_columns = icu_pat.columns.str.contains('AGE|GENDER_M|LOS|NUM_RECENT_ADMISSIONS|ADMISSION_LOCATION|INSURANCE|MARITAL_STATUS|ETHNICITY|PRE_ICU_LOS|ELECTIVE_SURGERY|VALUECAT')
  static = icu_pat.loc[:, static_columns].values
  static_vars = icu_pat.loc[:, static_columns].columns.values.tolist()
  
  # classification label
  print('Create label array...')
  label = icu_pat.loc[:, 'POSITIVE'].values
  
  print('-----------------------------------------')
  
  print('Split data into train/validate/test...')
  # Split patients to avoid data leaks
  patients = icu_pat['SUBJECT_ID'].drop_duplicates()
  train, validate, test = np.split(patients.sample(frac=1, random_state=42), [int(.9*len(patients)), int(.9*len(patients))])
  train_ids = icu_pat['SUBJECT_ID'].isin(train).values
  test_ids = icu_pat['SUBJECT_ID'].isin(test).values

  data_train = static[train_ids, :]
  data_test = static[test_ids, :]
  
  label_train = label[train_ids]
  label_test = label[test_ids]  

  print('-----------------------------------------')  
  
  # Fit logistic regression model
  print('Fit logistic regression model...')
  regr = linear_model.LogisticRegression()
  regr.fit(data_train, label_train)
  label_sigmoids = regr.predict_proba(data_test)[:, 1]

  print('Evaluate...')
  # Average precision
  avpre = average_precision_score(label_test, label_sigmoids)
  
  # Determine AUROC score
  auroc = roc_auc_score(label_test, label_sigmoids)
  
  # Sensitivity, specificity
  fpr, tpr, thresholds = roc_curve(label_test, label_sigmoids)
  youden_idx = np.argmax(tpr - fpr)
  sensitivity = tpr[youden_idx]
  specificity = 1-fpr[youden_idx]

  # F1 score
  f1_final = 0
  for t in thresholds:
    label_pred = (np.array(label_sigmoids) >= t).astype(int)
    f1 = f1_score(label_test, label_pred)
    if f1 > f1_final:
      f1_final = f1
  
  # Calibration
  fraction_of_positives, mean_predicted_value = calibration_curve(label_test, label_sigmoids, n_bins=10)
  plt.plot([0, 1], [0, 1], 'k:')
  plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
  ax = plt.gca()
  ax.set_xlabel('Mean Predicted Probability')
  ax.set_ylabel('True Probability (Fractions of Positives)')
  ax.set_ylim([-0.05, 1.05])
  ax.legend(loc='upper left')
  ax.set_title('Calibration plots (reliability curve)')
  plt.show()

  print('------------------------------------------------')
  print('Average Precision score: {}'.format(round(avpre)))
  print('AUROC score: {}'.format(round(auroc)))
  print('Sensitivity: {}'.format(round(sensitivity)))
  print('Specificity: {}'.format(round(specificity)))
  print('F1: {}'.format(round(f1_final)))
  print('Done')
  

