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
  print('Load charts and outputs...')
  charts_outputs = pd.read_pickle(hp.data_dir + 'charts_outputs_reduced.pkl')

  print('-----------------------------------------')
  print('Load prescriptions...')
  dtype = {'ICUSTAY_ID': 'str',
           'DRUG': 'str',
           'STARTDATE': 'str'}
  parse_dates = ['STARTDATE']
  # Load prescriptions table
  # Table purpose: Contains medication related order entries, i.e. prescriptions
  prescriptions = pd.read_csv(hp.mimic_dir + 'PRESCRIPTIONS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
  prescriptions = prescriptions.dropna()
  prescriptions['ICUSTAY_ID'] = prescriptions['ICUSTAY_ID'].astype('int32')
  prescriptions['DRUG'] = 'PRESC_' + prescriptions['DRUG'].str.lower().replace('\s+', '', regex=True)
  prescriptions = prescriptions.rename(columns={'DRUG': 'VALUECAT', 'STARTDATE': 'CHARTTIME'})
  df = pd.concat([charts_outputs, prescriptions], ignore_index=True, sort=False)
    
  print('-----------------------------------------')

  # Link charts/outputs and icu_pat tables
  print('Link charts/outputs and icu_pat tables...')
  df = pd.merge(icu_pat[['ICUSTAY_ID', 'OUTTIME']], df, how='left', on=['ICUSTAY_ID'])
  
  # Reset time value using time difference to DISCHTIME (0 if negative)
  df['HOURS_TO_OUT'] = (df['OUTTIME']-df['CHARTTIME']) / np.timedelta64(1, 'h')
  df.loc[df['HOURS_TO_OUT'] < 0, 'HOURS_TO_OUT'] = 0
  df = df.drop(columns=['OUTTIME', 'CHARTTIME'])

  print('Drop duplicates...')
  df = df.drop_duplicates()

  print('Map rare codes to OTHER...')
  df = df.apply(lambda x: x.mask(x.map(x.value_counts()) < hp.min_count, 'other') if x.name in ['VALUECAT'] else x)                   
  
  print('-----------------------------------------')  
  print('Save...')
  assert len(df['ICUSTAY_ID'].unique()) == 45298
  df.sort_values(by=['ICUSTAY_ID', 'HOURS_TO_OUT'], ascending=[True, True], inplace=True)
  df.to_pickle(hp.data_dir + 'charts_prescriptions.pkl')
  df.to_csv(hp.data_dir + 'charts_prescriptions.csv', index=False)

    
