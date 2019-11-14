from hyperparameters import Hyperparameters as hp
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from pdb import set_trace as bp

pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
  # Load icu_pat table
  print('Loading icu_pat...')
  icu_pat = pd.read_pickle(hp.data_dir + 'icu_pat_admit.pkl')
  
  print('-----------------------------------------')
  print('Load admissions...')
  # Load admissions table
  # Table purpose: Define a patients hospital admission, HADM_ID.
  dtype = {'HADM_ID': 'int32',
           'ADMITTIME': 'str',
           'DISCHTIME': 'str'}
  parse_dates = ['ADMITTIME', 'DISCHTIME']
  admissions = pd.read_csv(hp.mimic_dir + 'ADMISSIONS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
  
  print('-----------------------------------------')
  print('Load diagnoses and procedures...')
  dtype = {'SUBJECT_ID': 'int32',
           'HADM_ID': 'int32',
           'ICD9_CODE': 'str'}  
  # Load diagnosis_icd table
  # Table purpose: Contains ICD diagnoses for patients, most notably ICD-9 diagnoses.
  diagnoses = pd.read_csv(hp.mimic_dir + 'DIAGNOSES_ICD.csv', usecols=dtype.keys(), dtype=dtype)
  diagnoses = diagnoses.dropna()
  # Load procedures_icd table
  # Table purpose: Contains ICD procedures for patients, most notably ICD-9 procedures.
  procedures = pd.read_csv(hp.mimic_dir + 'PROCEDURES_ICD.csv', usecols=dtype.keys(), dtype=dtype)
  procedures = procedures.dropna()
  
  # Merge diagnoses and procedures
  diagnoses['ICD9_CODE'] = 'DIAGN_' + diagnoses['ICD9_CODE'].str.lower().str.strip()
  procedures['ICD9_CODE'] = 'PROCE_' + procedures['ICD9_CODE'].str.lower().str.strip()
  diag_proc = pd.concat([diagnoses, procedures], ignore_index=True, sort=False)
  
  print('-----------------------------------------')
    
  # Link diagnoses/procedures and admissions tables
  print('Link diagnoses/procedures and admissions tables...')
  diag_proc = pd.merge(diag_proc, admissions, how='inner', on='HADM_ID').drop(columns=['HADM_ID'])

  # Link diagnoses/procedures and icu_pat tables
  print('Link diagnoses/procedures and icu_pat tables...')
  diag_proc = pd.merge(icu_pat[['SUBJECT_ID', 'ICUSTAY_ID', 'OUTTIME']], diag_proc, how='left', on=['SUBJECT_ID'])
  
  # Remove codes related to future admissions using time difference to ADMITTIME
  diag_proc['DAYS_TO_OUT'] = (diag_proc['OUTTIME']-diag_proc['ADMITTIME']) / np.timedelta64(1, 'D')
  diag_proc = diag_proc[(diag_proc['DAYS_TO_OUT'] >= 0) | diag_proc['DAYS_TO_OUT'].isna()]
  # Reset time value using time difference to DISCHTIME (0 if negative)
  diag_proc['DAYS_TO_OUT'] = (diag_proc['OUTTIME']-diag_proc['DISCHTIME']) / np.timedelta64(1, 'D')
  diag_proc.loc[diag_proc['DAYS_TO_OUT'] < 0, 'DAYS_TO_OUT'] = 0
  diag_proc = diag_proc.drop(columns=['SUBJECT_ID', 'OUTTIME', 'ADMITTIME', 'DISCHTIME'])
  # Lost some ICUSTAY_IDs with only negative DAYS_TO_OUT, merge back
  diag_proc = pd.merge(icu_pat[['ICUSTAY_ID']], diag_proc, how='left', on=['ICUSTAY_ID'])
  
  print('Drop duplicates...')
  diag_proc = diag_proc.drop_duplicates()

  print('Map rare codes to OTHER...')
  diag_proc = diag_proc.apply(lambda x: x.mask(x.map(x.value_counts()) < hp.min_count, 'other') if x.name in ['ICD9_CODE'] else x)                   
  
  print('-----------------------------------------')  
  print('Save...')
  assert len(diag_proc['ICUSTAY_ID'].unique()) == 45298
  diag_proc.sort_values(by=['ICUSTAY_ID', 'DAYS_TO_OUT'], ascending=[True, True], inplace=True)
  diag_proc.to_pickle(hp.data_dir + 'diag_proc.pkl')
  diag_proc.to_csv(hp.data_dir + 'diag_proc.csv', index=False)
    
