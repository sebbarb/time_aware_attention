from hyperparameters import Hyperparameters as hp
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from pdb import set_trace as bp

pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == '__main__':
  # Load icustays table
  # Table purpose: Defines each ICUSTAY_ID in the database, i.e. defines a single ICU stay
  print('Load ICU stays...')
  dtype = {'SUBJECT_ID': 'int32',
           'HADM_ID': 'int32',
           'ICUSTAY_ID': 'int32',
           'INTIME': 'str',
           'OUTTIME': 'str',
           'LOS': 'float32'}
  parse_dates = ['INTIME', 'OUTTIME']
  icustays = pd.read_csv(hp.mimic_dir + 'ICUSTAYS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

  print('-----------------------------------------')
  
  # Load patients table
  # Table purpose: Contains all charted data for all patients.
  print('Load patients...')
  dtype = {'SUBJECT_ID': 'int32',
           'GENDER': 'str',
           'DOB': 'str',
           'DOD': 'str'}
  parse_dates = ['DOB', 'DOD']
  patients = pd.read_csv(hp.mimic_dir + 'PATIENTS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)  
  
  # Adjust shifted DOBs for older patients (median imputation)
  old_patient = patients['DOB'].dt.year < 2000
  date_offset = pd.DateOffset(years=(300-91), days=(-0.4*365))
  patients['DOB'][old_patient] = patients['DOB'][old_patient].apply(lambda x: x + date_offset)

  # Replace GENDER by dummy binary column 
  patients = pd.get_dummies(patients, columns = ['GENDER'], drop_first=True)
  
  print('-----------------------------------------')
  print('Load admissions...')
  # Load admissions table
  # Table purpose: Define a patients hospital admission, HADM_ID.
  dtype = {'SUBJECT_ID': 'int32', 
           'HADM_ID': 'int32',
           'ADMISSION_LOCATION': 'str',
           'INSURANCE': 'str',
           'MARITAL_STATUS': 'str',
           'ETHNICITY': 'str',
           'ADMITTIME': 'str',
           'ADMISSION_TYPE': 'str'}
  parse_dates = ['ADMITTIME']
  admissions = pd.read_csv(hp.mimic_dir + 'ADMISSIONS.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
  
  print('-----------------------------------------')
  print('Load services...')
  # Load services table
  # Table purpose: Lists services that a patient was admitted/transferred under.
  dtype = {'SUBJECT_ID': 'int32', 
           'HADM_ID': 'int32',
           'TRANSFERTIME': 'str',
           'CURR_SERVICE': 'str'}
  parse_dates = ['TRANSFERTIME']
  services = pd.read_csv(hp.mimic_dir + 'SERVICES.csv', usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
  
  print('-----------------------------------------')
  
  # Link icustays and patients tables
  print('Link icustays and patients tables...')
  icu_pat = pd.merge(icustays, patients, how='inner', on='SUBJECT_ID')
  icu_pat.sort_values(by=['SUBJECT_ID', 'OUTTIME'], ascending=[True, False], inplace=True)
  assert len(icu_pat['SUBJECT_ID'].unique()) == 46476
  assert len(icu_pat['ICUSTAY_ID'].unique()) == 61532

  # Exclude icu stays during which patient died
  icu_pat = icu_pat[~(icu_pat['DOD'] <= icu_pat['OUTTIME'])]
  assert len(icu_pat['SUBJECT_ID'].unique()) == 43126
  assert len(icu_pat['ICUSTAY_ID'].unique()) == 56745
    
  # Determine number of icu discharges in the last 365 days
  print('Compute number of recent admissions...')
  icu_pat['NUM_RECENT_ADMISSIONS'] = 0
  for name, group in tqdm(icu_pat.groupby(['SUBJECT_ID'])):
    for index, row in group.iterrows():
      days_diff = (row['OUTTIME']-group['OUTTIME']).dt.days
      icu_pat.at[index, 'NUM_RECENT_ADMISSIONS'] = len(group[(days_diff > 0) & (days_diff <=365)])
  
  # Create age variable and exclude patients < 18 y.o.
  icu_pat['AGE'] = (icu_pat['OUTTIME'] - icu_pat['DOB']).dt.days/365.
  icu_pat = icu_pat[icu_pat['AGE'] >= 18]
  assert len(icu_pat['SUBJECT_ID'].unique()) == 35233
  assert len(icu_pat['ICUSTAY_ID'].unique()) == 48616

  # Time to next admission (discharge to admission!)
  icu_pat['DAYS_TO_NEXT'] = (icu_pat.groupby(['SUBJECT_ID']).shift(1)['INTIME'] - icu_pat['OUTTIME']).dt.days

  # Add early readmission flag (less than 30 days after discharge)
  icu_pat['POSITIVE'] = (icu_pat['DAYS_TO_NEXT'] <= 30)
  assert icu_pat['POSITIVE'].sum() == 5495

  # Add early death flag (less than 30 days after discharge)
  early_death = ((icu_pat['DOD'] - icu_pat['OUTTIME']).dt.days <= 30)
  assert early_death.sum() == 3795
  
  # Censor negative patients who died within less than 30 days after discharge (no chance of readmission)
  icu_pat = icu_pat[icu_pat['POSITIVE'] | ~early_death]
  assert len(icu_pat['SUBJECT_ID'].unique()) == 33150
  assert len(icu_pat['ICUSTAY_ID'].unique()) == 45298
  
  # Clean up
  icu_pat.drop(columns=['DOB', 'DOD', 'DAYS_TO_NEXT'], inplace=True)
  
  print('-----------------------------------------')
  
  # Link icu_pat and admissions tables
  print('Link icu_pat and admissions tables...')
  icu_pat_admit = pd.merge(icu_pat, admissions, how='left', on=['SUBJECT_ID', 'HADM_ID'])
  print(icu_pat_admit.isnull().sum())

  print('Some data cleaning on admissions...')
  icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('WHITE'), 'ETHNICITY']    = 'WHITE'
  icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('BLACK'), 'ETHNICITY']    = 'BLACK/AFRICAN AMERICAN'
  icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('ASIAN'), 'ETHNICITY']    = 'ASIAN'
  icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('HISPANIC'), 'ETHNICITY'] = 'HISPANIC/LATINO'
  icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('DECLINED'), 'ETHNICITY'] = 'OTHER/UNKNOWN'
  icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('MULTI'), 'ETHNICITY']    = 'OTHER/UNKNOWN'
  icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('UNKNOWN'), 'ETHNICITY']  = 'OTHER/UNKNOWN'
  icu_pat_admit.loc[icu_pat_admit['ETHNICITY'].str.contains('OTHER'), 'ETHNICITY']  = 'OTHER/UNKNOWN'
  
  icu_pat_admit['MARITAL_STATUS'].fillna('UNKNOWN', inplace=True)
  icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('MARRIED'), 'MARITAL_STATUS']      = 'MARRIED/LIFE PARTNER'
  icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('LIFE PARTNER'), 'MARITAL_STATUS'] = 'MARRIED/LIFE PARTNER'
  icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('WIDOWED'), 'MARITAL_STATUS']      = 'WIDOWED/DIVORCED/SEPARATED'
  icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('DIVORCED'), 'MARITAL_STATUS']     = 'WIDOWED/DIVORCED/SEPARATED'
  icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('SEPARATED'), 'MARITAL_STATUS']    = 'WIDOWED/DIVORCED/SEPARATED'
  icu_pat_admit.loc[icu_pat_admit['MARITAL_STATUS'].str.contains('UNKNOWN'), 'MARITAL_STATUS']      = 'OTHER/UNKNOWN'
  
  columns_to_mask = ['ADMISSION_LOCATION',
                     'INSURANCE',
                     'MARITAL_STATUS',
                     'ETHNICITY']
  icu_pat_admit = icu_pat_admit.apply(lambda x: x.mask(x.map(x.value_counts()) < hp.min_count, 'OTHER/UNKNOWN') if x.name in columns_to_mask else x)                   
  icu_pat_admit = icu_pat_admit.apply(lambda x: x.str.title() if x.name in columns_to_mask else x)
  
  # Compute pre-ICU length of stay in fractional days
  icu_pat_admit['PRE_ICU_LOS'] = (icu_pat_admit['INTIME'] - icu_pat_admit['ADMITTIME']) / np.timedelta64(1, 'D')
  icu_pat_admit.loc[icu_pat_admit['PRE_ICU_LOS']<0, 'PRE_ICU_LOS'] = 0
  
  # Clean up
  icu_pat_admit.drop(columns=['ADMITTIME'], inplace=True)

  print('-----------------------------------------')
  
  # Link services table
  # Keep first service only
  services.sort_values(by=['HADM_ID', 'TRANSFERTIME'], ascending=True, inplace=True)
  services = services.groupby(['HADM_ID']).nth(0).reset_index()
  
  # Check if first service is a surgery
  services['SURGERY'] = services['CURR_SERVICE'].str.contains('SURG') | (services['CURR_SERVICE'] == 'ORTHO')
  
  print('Link services table...')  
  icu_pat_admit = pd.merge(icu_pat_admit, services, how='left', on=['SUBJECT_ID', 'HADM_ID'])
  
  # Get elective surgery admissions
  icu_pat_admit['ELECTIVE_SURGERY'] = ((icu_pat_admit['ADMISSION_TYPE'] == 'ELECTIVE') & icu_pat_admit['SURGERY']).astype(int)

  # Clean up
  icu_pat_admit.drop(columns=['TRANSFERTIME', 'CURR_SERVICE', 'ADMISSION_TYPE', 'SURGERY'], inplace=True)

  print('-----------------------------------------')
  # Baseline characteristics table
  pos = icu_pat_admit[icu_pat_admit['POSITIVE']==1]
  neg = icu_pat_admit[icu_pat_admit['POSITIVE']==0]
  print('Total pos {}'.format(len(pos)))
  print('Total neg {}'.format(len(neg)))
  print(pos['LOS'].describe())
  print(neg['LOS'].describe())
  print((pos['PRE_ICU_LOS']).describe())
  print((neg['PRE_ICU_LOS']).describe())  
  pd.set_option('precision', 1)
  print(pos['AGE'].describe())
  print(neg['AGE'].describe())
  print(pos['NUM_RECENT_ADMISSIONS'].describe())
  print(neg['NUM_RECENT_ADMISSIONS'].describe())
  print(pd.DataFrame({'COUNTS': pos['GENDER_M'].value_counts(), 'PERC': pos['GENDER_M'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': neg['GENDER_M'].value_counts(), 'PERC': neg['GENDER_M'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': pos['ADMISSION_LOCATION'].value_counts(), 'PERC': pos['ADMISSION_LOCATION'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': neg['ADMISSION_LOCATION'].value_counts(), 'PERC': neg['ADMISSION_LOCATION'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': pos['INSURANCE'].value_counts(), 'PERC': pos['INSURANCE'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': neg['INSURANCE'].value_counts(), 'PERC': neg['INSURANCE'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': pos['MARITAL_STATUS'].value_counts(), 'PERC': pos['MARITAL_STATUS'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': neg['MARITAL_STATUS'].value_counts(), 'PERC': neg['MARITAL_STATUS'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': pos['ETHNICITY'].value_counts(), 'PERC': pos['ETHNICITY'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': neg['ETHNICITY'].value_counts(), 'PERC': neg['ETHNICITY'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': pos['ELECTIVE_SURGERY'].value_counts(), 'PERC': pos['ELECTIVE_SURGERY'].value_counts(normalize=True)*100}))
  print(pd.DataFrame({'COUNTS': neg['ELECTIVE_SURGERY'].value_counts(), 'PERC': neg['ELECTIVE_SURGERY'].value_counts(normalize=True)*100}))  
  print('-----------------------------------------')
  
  print('Save...')
  assert len(icu_pat_admit) == 45298
  icu_pat_admit.sort_values(by='ICUSTAY_ID', ascending=True, inplace=True)
  icu_pat_admit.to_pickle(hp.data_dir + 'icu_pat_admit.pkl')
  icu_pat_admit.to_csv(hp.data_dir + 'icu_pat_admit.csv', index=False)
    
