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
  print('Loading data...')
  icu_pat = pd.read_pickle(hp.data_dir + 'icu_pat_admit.pkl')
  
  print('Loading last vital signs measurements...')
  charts = pd.read_pickle(hp.data_dir + 'charts_outputs_last_only.pkl')
  charts = charts.drop(columns=['CHARTTIME'])
  charts = pd.get_dummies(charts, columns = ['VALUECAT']).groupby('ICUSTAY_ID').sum()
  charts.drop(columns=['VALUECAT_CHART_BP_n', 'VALUECAT_CHART_BT_n', 'VALUECAT_CHART_GC_n', 'VALUECAT_CHART_HR_n', 'VALUECAT_CHART_RR_n', 'VALUECAT_CHART_UO_n'], inplace=True) # drop reference columns
  
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
  train, validate, test = np.split(patients.sample(frac=1, random_state=123), [int(.9*len(patients)), int(.9*len(patients))])
  train_ids = icu_pat['SUBJECT_ID'].isin(train).values
  test_ids = icu_pat['SUBJECT_ID'].isin(test).values

  data_train = static[train_ids, :]
  data_test = static[test_ids, :]
  
  label_train = label[train_ids]
  label_test = label[test_ids]  
  
  # Patients in test data
  test_ids_patients = pd.read_pickle(hp.data_dir + 'test_ids_patients.pkl')
  patients = test_ids_patients.drop_duplicates()
  num_patients = patients.shape[0]
  row_ids = pd.DataFrame({'ROW_IDX': test_ids_patients.index}, index=test_ids_patients)

  print('-----------------------------------------')  
  
  # Fit logistic regression model
  print('Fit logistic regression model...')
  regr = linear_model.LogisticRegression()
  regr.fit(data_train, label_train)
  
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

    sample_patients = patients.sample(n=num_patients, replace=True)
    idx = np.squeeze(row_ids.loc[sample_patients].values)
    data_test_bs, label_test_bs = data_test[idx], label_test[idx]
  
    label_sigmoids = regr.predict_proba(data_test_bs)[:, 1]

    print('Evaluate...')
    # Average precision
    avpre = average_precision_score(label_test_bs, label_sigmoids)
    
    # Determine AUROC score
    auroc = roc_auc_score(label_test_bs, label_sigmoids)
    
    # Sensitivity, specificity
    fpr, tpr, thresholds = roc_curve(label_test_bs, label_sigmoids)
    youden_idx = np.argmax(tpr - fpr)
    sensitivity = tpr[youden_idx]
    specificity = 1-fpr[youden_idx]

    # F1, PPV, NPV score
    f1 = 0
    ppv = 0
    npv = 0
    for t in thresholds:
      label_pred = (np.array(label_sigmoids) >= t).astype(int)
      f1_temp = f1_score(label_test_bs, label_pred)
      ppv_temp = precision_score(label_test_bs, label_pred, pos_label=1)
      npv_temp = precision_score(label_test_bs, label_pred, pos_label=0)
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
  
  print('------------------------------------------------')
  print('Net variant: logistic regression')
  print('Average Precision: {} [{},{}]'.format(round(avpre_mean), round(avpre_lci), round(avpre_uci)))
  print('AUROC: {} [{},{}]'.format(round(auroc_mean), round(auroc_lci), round(auroc_uci)))
  print('F1: {} [{},{}]'.format(round(f1_mean), round(f1_lci), round(f1_uci)))  
  print('PPV: {} [{},{}]'.format(round(ppv_mean), round(ppv_lci), round(ppv_uci)))
  print('NPV: {} [{},{}]'.format(round(npv_mean), round(npv_lci), round(npv_uci)))
  print('Sensitivity: {} [{},{}]'.format(round(sensitivity_mean), round(sensitivity_lci), round(sensitivity_uci)))
  print('Specificity: {} [{},{}]'.format(round(specificity_mean), round(specificity_lci), round(specificity_uci)))
  print('Done')
  

