from hyperparameters import Hyperparameters as hp
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from pdb import set_trace as bp

pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == '__main__':
  # Load (reduced) chartevents table
  print('Loading chart events...')
  dtype = {'SUBJECT_ID': 'int32',
           'ICUSTAY_ID': 'int32',
           'CE_TYPE': 'str',
           'CHARTTIME': 'str',
           'VALUENUM': 'float32'}
  parse_dates = ['CHARTTIME']
  charts = pd.read_csv(hp.data_dir + 'chartevents_reduced.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

  print('-----------------------------------------')

  print('Compute BMI and GCS total...')
  charts.sort_values(by=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], ascending=[True, True, False], inplace=True)
  
  # Compute BMI
  rows_bmi = (charts['CE_TYPE']=='WEIGHT') | (charts['CE_TYPE']=='HEIGHT')
  charts_bmi = charts[rows_bmi]
  charts_bmi = charts_bmi.pivot_table(index=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], columns='CE_TYPE', values='VALUENUM')
  charts_bmi = charts_bmi.rename_axis(None, axis=1).reset_index()
  charts_bmi['HEIGHT'] = charts_bmi.groupby('SUBJECT_ID')['HEIGHT'].ffill()
  charts_bmi['HEIGHT'] = charts_bmi.groupby('SUBJECT_ID')['HEIGHT'].bfill()
  charts_bmi =  charts_bmi[~pd.isnull(charts_bmi).any(axis=1)]
  charts_bmi['VALUENUM'] = charts_bmi['WEIGHT']/charts_bmi['HEIGHT']/charts_bmi['HEIGHT']*10000
  charts_bmi['CE_TYPE'] = 'BMI'
  charts_bmi.drop(columns=['HEIGHT', 'WEIGHT'], inplace=True)
  
  # Compute GCS total if not available
  rows_gcs = (charts['CE_TYPE']=='GCS_EYE_OPENING') | (charts['CE_TYPE']=='GCS_VERBAL_RESPONSE') | (charts['CE_TYPE']=='GCS_MOTOR_RESPONSE') | (charts['CE_TYPE']=='GCS_TOTAL')
  charts_gcs = charts[rows_gcs]
  charts_gcs = charts_gcs.pivot_table(index=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], columns='CE_TYPE', values='VALUENUM')
  charts_gcs = charts_gcs.rename_axis(None, axis=1).reset_index()
  null_gcs_total = charts_gcs['GCS_TOTAL'].isnull()
  charts_gcs.loc[null_gcs_total, 'GCS_TOTAL'] = charts_gcs[null_gcs_total].GCS_EYE_OPENING + charts_gcs[null_gcs_total].GCS_VERBAL_RESPONSE + charts_gcs[null_gcs_total].GCS_MOTOR_RESPONSE
  charts_gcs =  charts_gcs[~charts_gcs['GCS_TOTAL'].isnull()]
  charts_gcs = charts_gcs.rename(columns={'GCS_TOTAL': 'VALUENUM'})
  charts_gcs['CE_TYPE'] = 'GCS_TOTAL'
  charts_gcs.drop(columns=['GCS_EYE_OPENING', 'GCS_VERBAL_RESPONSE', 'GCS_MOTOR_RESPONSE'], inplace=True)

  # Merge back with rest of the table
  rows_others = ~rows_bmi & ~rows_gcs
  charts = pd.concat([charts_bmi, charts_gcs, charts[rows_others]], ignore_index=True, sort=False)
  charts.drop(columns=['SUBJECT_ID'], inplace=True)
  charts.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)

  print('-----------------------------------------')

  # Load (reduced) outputevents table
  print('Loading output events...')
  outputs = pd.read_pickle(hp.data_dir + 'outputevents_reduced.pkl')
  df = pd.concat([charts, outputs], ignore_index=True, sort=False)
  df.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)

  print('-----------------------------------------')

  print('Create categorical variable...')
  # Bin according to OASIS severity score
  heart_rate_bins               = np.array([-1, 32.99, 88.5, 106.5, 125.5, np.Inf])
  respiratory_rate_bins         = np.array([-1, 5.99, 12.5, 22.5, 30.5, 44.5, np.Inf])
  body_temperature_bins         = np.array([-1, 33.21, 35.93, 36.39, 36.88, 39.88, np.Inf])
  mean_bp_bins                  = np.array([-1, 20.64, 50.99, 61.32, 143.44, np.Inf])
  fraction_inspired_oxygen_bins = np.array([-1, np.Inf])
  gcs_total_bins                = np.array([-1, 7, 13, 14, 15])
  bmi_bins                      = np.array([-1, 15, 16, 18.5, 25, 30, 35, 40, 45, 50, 60, np.Inf])
  urine_output_bins             = np.array([-1, 670.99, 1426.99, 2543.99, 6896, np.Inf])
  #bins = [heart_rate_bins, respiratory_rate_bins, body_temperature_bins, mean_bp_bins, fraction_inspired_oxygen_bins, gcs_total_bins, bmi_bins, urine_output_bins]
  bins = [heart_rate_bins, respiratory_rate_bins, body_temperature_bins, mean_bp_bins, fraction_inspired_oxygen_bins, gcs_total_bins, urine_output_bins]
  # Labels 
  heart_rate_labels               = ['CHART_HR_m1', 'CHART_HR_n', 'CHART_HR_p1', 'CHART_HR_p2', 'CHART_HR_p3']
  respiratory_rate_labels         = ['CHART_RR_m2', 'CHART_RR_m1', 'CHART_RR_n', 'CHART_RR_p1', 'CHART_RR_p2', 'CHART_RR_p3']
  body_temperature_labels         = ['CHART_BT_m3', 'CHART_BT_m2', 'CHART_BT_m1', 'CHART_BT_n', 'CHART_BT_p1', 'CHART_BT_p2']
  mean_bp_labels                  = ['CHART_BP_m3', 'CHART_BP_m2', 'CHART_BP_m1', 'CHART_BP_n', 'CHART_BP_p1']
  fraction_inspired_oxygen_labels = ['CHART_VENT']
  gcs_total_labels                = ['CHART_GC_m3', 'CHART_GC_m2', 'CHART_GC_m1', 'CHART_GC_n']
  bmi_labels                      = ['CHART_BM_m3', 'CHART_BM_m2', 'CHART_BM_m1', 'CHART_BM_n', 'CHART_BM_p1', 'CHART_BM_p2', 'CHART_BM_p3', 'CHART_BM_p4', 'CHART_BM_p5', 'CHART_BM_p6', 'CHART_BM_p7']
  urine_output_labels             = ['CHART_UO_m3', 'CHART_UO_m2', 'CHART_UO_m1', 'CHART_UO_n', 'CHART_UO_p1']
  #labels = [heart_rate_labels, respiratory_rate_labels, body_temperature_labels, mean_bp_labels, fraction_inspired_oxygen_labels, gcs_total_labels, bmi_labels, urine_output_labels]
  labels = [heart_rate_labels, respiratory_rate_labels, body_temperature_labels, mean_bp_labels, fraction_inspired_oxygen_labels, gcs_total_labels, urine_output_labels]
  # Chart event types
  #ce_types = ['HEART_RATE', 'RESPIRATORY_RATE', 'BODY_TEMPERATURE', 'MEAN_BP', 'FRACTION_INSPIRED_OXYGEN', 'GCS_TOTAL', 'BMI', 'URINE_OUTPUT']
  ce_types = ['HEART_RATE', 'RESPIRATORY_RATE', 'BODY_TEMPERATURE', 'MEAN_BP', 'FRACTION_INSPIRED_OXYGEN', 'GCS_TOTAL', 'URINE_OUTPUT']
  
  df_list = []
  df_list_last_only = [] # for logistic regression
  for type, label, bin in zip(ce_types, labels, bins):
    # get chart events of a specific type
    tmp = df[df['CE_TYPE'] == type]
    # bin them and sort
    tmp['VALUECAT'] = pd.cut(tmp['VALUENUM'], bins=bin, labels=label)
    tmp.drop(columns=['CE_TYPE', 'VALUENUM'], inplace=True)
    tmp.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)
    # remove consecutive duplicates
    tmp = tmp[(tmp[['ICUSTAY_ID', 'VALUECAT']] != tmp[['ICUSTAY_ID', 'VALUECAT']].shift()).any(axis=1)]
    df_list.append(tmp)
    # for logistic regression, keep only the last measurement
    tmp = tmp.drop_duplicates(subset='ICUSTAY_ID')
    df_list_last_only.append(tmp)
  
  df = pd.concat(df_list, ignore_index=True, sort=False)
  df.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)
  
  # drop duplicates to keep size manageable
  df = df.drop_duplicates()
  
  print('-----------------------------------------')
  
  print('Save...')
  df.to_pickle(hp.data_dir + 'charts_outputs_reduced.pkl')
  df.to_csv(hp.data_dir + 'charts_outputs_reduced.csv', index=False)

  print('-----------------------------------------')
  
  print('Save data for logistic regression...')

  # for logistic regression
  df_last_only = pd.concat(df_list_last_only, ignore_index=True, sort=False)
  df_last_only.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False], inplace=True)
  df_last_only.to_pickle(hp.data_dir + 'charts_outputs_last_only.pkl')
  df_last_only.to_csv(hp.data_dir + 'charts_outputs_last_only.csv', index=False)  
    
