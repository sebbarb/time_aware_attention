from hyperparameters import Hyperparameters as hp
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from pdb import set_trace as bp

pd.options.mode.chained_assignment = None  # default='warn'

# Relevant ITEMIDs, from https://github.com/vincentmajor/mimicfilters/blob/master/lists/OASIS_components/preprocess_urine_awk_str.txt
urine_output = [42810, 43171, 43173, 43175, 43348, 43355, 43365, 43372, 43373, 43374, 43379, 43380, 43431, 43462, 43522, 40405, 40428, 40534, 
40288, 42042, 42068, 42111, 42119, 42209, 41857, 40715, 40056, 40061, 40085, 40094, 40096, 42001, 42676, 42556, 43093, 44325, 44706,
44506, 42859, 44237, 44313, 44752, 44824, 44837, 43576, 43589, 43633, 44911, 44925, 42362, 42463, 42507, 42510, 40055, 40057, 40065,
40069, 45804, 45841, 43811, 43812, 43856, 43897, 43931, 43966, 44080, 44103, 44132, 45304, 46177, 46532, 46578, 46658, 46748, 40651,
43053, 43057, 40473, 42130, 41922, 44253, 44278, 46180, 44684, 43333, 43347, 42592, 42666, 42765, 42892, 45927, 44834, 43638, 43654,
43519, 43537, 42366, 45991, 46727, 46804, 43987, 44051, 227489, 226566, 226627, 226631, 45415, 42111, 41510, 40055, 226559, 40428,
40580, 40612, 40094, 40848, 43685, 42362, 42463, 42510, 46748, 40972, 40973, 46456, 226561, 226567, 226632, 40096, 40651, 226557,
226558, 40715, 226563]

if __name__ == '__main__':
  # Relevant ITEMIDs
  print('-----------------------------------------')
  print('Load item definitions')
  dtype = {'ITEMID': 'int32',
           'LABEL': 'str',
           'UNITNAME': 'str',
           'LINKSTO': 'str'}
  defs = pd.read_csv(hp.mimic_dir + 'D_ITEMS.csv', usecols=dtype.keys(), dtype=dtype)
  print('URINE_OUTPUT')
  defs = defs[defs['ITEMID'].isin(urine_output)]
  defs['LABEL'] = defs['LABEL'].str.lower()
  # Remove measurements in /kg/hr
  defs = defs[~(defs['LABEL'].str.contains('hr') | defs['LABEL'].str.contains('kg')) | defs['LABEL'].str.contains('nephro')]
  print(defs['LABEL'])
  urine_output = defs['ITEMID'].tolist()
  print('-----------------------------------------')

  print('Loading Output Events')
  dtype = {'ICUSTAY_ID': 'str',
           'ITEMID': 'int32',
           'CHARTTIME': 'str',
           'VALUE': 'float32'}
  parse_dates = ['CHARTTIME']

  # Load outputevents table
  # Table purpose: Output data for patients.
  df = pd.read_csv(hp.mimic_dir + 'OUTPUTEVENTS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
  df = df.rename(columns={'VALUE': 'VALUENUM'})
  df = df[df['ICUSTAY_ID'].notna() & df['VALUENUM'].notna() & (df['ITEMID'].isin(urine_output)) & (df['VALUENUM'] > 0)]
  df['ICUSTAY_ID'] = df['ICUSTAY_ID'].astype('int32')
  
  # remove implausible measurements
  df = df[~(df.VALUENUM > 10000)]
  
  # sum all outputs in one day
  df.drop(columns=['ITEMID'], inplace=True)
  df['CHARTTIME'] = df['CHARTTIME'].dt.date
  df = df.groupby(['ICUSTAY_ID', 'CHARTTIME']).sum()
  df['CE_TYPE'] = 'URINE_OUTPUT'
  df = df[~(df.VALUENUM > 10000)]
  
  print('Remove admission and discharge days (since data on urine output is incomplete)')
  # Load icustays table
  # Table purpose: Defines each ICUSTAY_ID in the database, i.e. defines a single ICU stay
  print('Load ICU stays...')
  dtype = {'ICUSTAY_ID': 'int32',
           'INTIME': 'str',
           'OUTTIME': 'str'}
  parse_dates = ['INTIME', 'OUTTIME']
  icustays = pd.read_csv(hp.mimic_dir + 'ICUSTAYS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
  icustays['INTIME'] = icustays['INTIME'].dt.date
  icustays['OUTTIME'] = icustays['OUTTIME'].dt.date
  
  # Merge
  tmp = icustays[['ICUSTAY_ID', 'INTIME']].drop_duplicates()
  tmp = tmp.rename(columns={'INTIME': 'CHARTTIME'})
  tmp['ID_IN'] = 1
  df = pd.merge(df, tmp, how='left', on=['ICUSTAY_ID', 'CHARTTIME'])
  tmp = icustays[['ICUSTAY_ID', 'OUTTIME']].drop_duplicates()
  tmp = tmp.rename(columns={'OUTTIME': 'CHARTTIME'})
  tmp['ID_OUT'] = 1
  df = pd.merge(df, tmp, how='left', on=['ICUSTAY_ID', 'CHARTTIME'])

  # Remove admission and discharge days
  df = df[df['ID_IN'].isnull() & df['ID_OUT'].isnull()]
  df.drop(columns=['ID_IN', 'ID_OUT'], inplace=True)

  # Add SUBJECT_ID and HADM_ID
  icustays.drop(columns=['INTIME', 'OUTTIME'], inplace=True)  
  df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME']) + pd.DateOffset(hours=12)
  
  # Save
  df.to_pickle(hp.data_dir + 'outputevents_reduced.pkl')
  
