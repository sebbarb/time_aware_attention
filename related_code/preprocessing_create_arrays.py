from hyperparameters import Hyperparameters as hp
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from pdb import set_trace as bp

pd.options.mode.chained_assignment = None  # default='warn'

def get_arrays(df, code_column, time_column, quantile=1):
  df['COUNT'] = df.groupby(['ICUSTAY_ID']).cumcount()
  df = df[df['COUNT'] < df.groupby(['ICUSTAY_ID']).size().quantile(q=quantile)]
  max_count_df = df['COUNT'].max()+1
  print('max_count {}'.format(max_count_df))
  multiindex_df = pd.MultiIndex.from_product([icu_pat['ICUSTAY_ID'], range(max_count_df)], names = ['ICUSTAY_ID', 'COUNT'])
  df = df.set_index(['ICUSTAY_ID', 'COUNT'])

  print('Reindex df...')
  df = df.reindex(multiindex_df).fillna(0)
  print('done')
  df_times = df[time_column].values.reshape((num_icu_stays, max_count_df))
  df[code_column] = df[code_column].astype('category')
  dict_df = dict(enumerate(df[code_column].cat.categories))
  df[code_column] = df[code_column].cat.codes
  df = df[code_column].values.reshape((num_icu_stays, max_count_df))

  return df, df_times, dict_df
  

if __name__ == '__main__':
  # Load icu_pat table
  print('Loading icu_pat...')
  icu_pat = pd.read_pickle(hp.data_dir + 'icu_pat_admit.pkl')
  
  print('Loading diagnoses/procedures...')
  dp = pd.read_pickle(hp.data_dir + 'diag_proc.pkl')

  print('Loading charts/prescriptions...')
  cp = pd.read_pickle(hp.data_dir + 'charts_prescriptions.pkl')

  print('-----------------------------------------')
  
  num_icu_stays = len(icu_pat['ICUSTAY_ID'])
  
  # static variables
  print('Create static array...')
  icu_pat = pd.get_dummies(icu_pat, columns = ['ADMISSION_LOCATION', 'INSURANCE', 'MARITAL_STATUS', 'ETHNICITY'])
  icu_pat.drop(columns=['ADMISSION_LOCATION_Emergency Room Admit', 'INSURANCE_Medicare', 'MARITAL_STATUS_Married/Life Partner', 'ETHNICITY_White'], inplace=True) # drop reference columns
  static_columns = icu_pat.columns.str.contains('AGE|GENDER_M|LOS|NUM_RECENT_ADMISSIONS|ADMISSION_LOCATION|INSURANCE|MARITAL_STATUS|ETHNICITY|PRE_ICU_LOS|ELECTIVE_SURGERY')
  static = icu_pat.loc[:, static_columns].values
  static_vars = icu_pat.loc[:, static_columns].columns.values.tolist()
  
  # classification label
  print('Create label array...')
  label = icu_pat.loc[:, 'POSITIVE'].values
  
  # diagnoses/procedures and charts/prescriptions
  print('Create diagnoses/procedures and charts/prescriptions array...')
  dp, dp_times, dict_dp = get_arrays(dp, 'ICD9_CODE', 'DAYS_TO_OUT', 1)
  cp, cp_times, dict_cp = get_arrays(cp, 'VALUECAT', 'HOURS_TO_OUT', 0.95)
  
  # Normalize times
  dp_times = dp_times/dp_times.max()
  cp_times = cp_times/cp_times.max()

  print('-----------------------------------------')
  
  print('Split data into train/validate/test...')
  # Split patients to avoid data leaks
  patients = icu_pat['SUBJECT_ID'].drop_duplicates()
  train, validate, test = np.split(patients.sample(frac=1, random_state=123), [int(.9*len(patients)), int(.9*len(patients))])
  train_ids = icu_pat['SUBJECT_ID'].isin(train).values
  validate_ids = icu_pat['SUBJECT_ID'].isin(validate).values
  test_ids = icu_pat['SUBJECT_ID'].isin(test).values
  
  print('Get patients corresponding to test ids')
  test_ids_patients = icu_pat['SUBJECT_ID'].iloc[test_ids].reset_index(drop=True)

  print('-----------------------------------------')
  
  print('Save...')
  # np.savez(hp.data_dir + 'data_arrays.npz', static=static, static_vars=static_vars, label=label,
           # dp=dp, cp=cp, dp_times=dp_times, cp_times=cp_times, dict_dp=dict_dp, dict_cp=dict_cp,
           # train_ids=train_ids, validate_ids=validate_ids, test_ids=test_ids)
  # np.savez(hp.data_dir + 'data_dictionaries.npz', dict_dp=dict_dp, dict_cp=dict_cp)
  test_ids_patients.to_pickle(hp.data_dir + 'test_ids_patients.pkl')
           



