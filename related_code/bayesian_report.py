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
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import rgb2hex
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from pdb import set_trace as bp

if __name__ == '__main__':
  # Load icu_pat table
  print('Loading icu_pat...')
  icu_pat = pd.read_pickle(hp.data_dir + 'icu_pat_admit.pkl')

  # Forward pass
  data = np.load(hp.data_dir + 'data_fp.npz')
  weights_diag = data['weights_diag']
  weights_proc = data['weights_proc']
  weights_pres = data['weights_pres']
  charts_activations = data['charts_activations']
  all = data['all']
  
  data = np.load(hp.data_dir + 'data_fp_samples.npz')
  label_sigmoids = data['label_sigmoids']
  pred_samples = data['pred_samples']  

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
  
  # Codes significance
  mean_coeff_diag = torch.matmul(net.embed_diag.weight.mu, net.fc_diag.weight.mu.t()).detach().cpu().numpy().squeeze()*mu[num_static]
  mean_coeff_proc = torch.matmul(net.embed_proc.weight.mu, net.fc_proc.weight.mu.t()).detach().cpu().numpy().squeeze()*mu[num_static+1]
  mean_coeff_pres = torch.matmul(net.embed_pres.weight.mu, net.fc_pres.weight.mu.t()).detach().cpu().numpy().squeeze()*mu[num_static+2]
  mean_coeff_diag[0], mean_coeff_proc[0], mean_coeff_pres[0] = 0, 0, 0
  
  chart_labels = np.array(['T [C]', 'DBP [mmHg]', 'FiO2', 'GCS Eye', 'GCS Motor', 'GCS Verbal', 'BG [mg/dL]', 'HR [BPM]', 'Sats [%]', 'pH', 'RR [/min]', 'SBP [mmHg]', 'BMI [kg/m$^2$]'])
  # bp()
  print('-----------------------------------------')
  all_avg_pred = np.mean(pred_samples, axis=1)
  all_ci_lower = np.percentile(pred_samples,  5, axis=1)
  all_ci_upper = np.percentile(pred_samples, 95, axis=1)
  
  for pat in tqdm(range(len(label))):
    avg_pred = int(100*all_avg_pred[pat])
    # if (92 <= avg_pred <= 92):
    if ((icu_pat.loc[pat, 'SUBJECT_ID'] == 17843) or  (icu_pat.loc[pat, 'SUBJECT_ID'] == 59543)):
      ci_lower = int(100*all_ci_lower[pat])
      ci_upper = int(100*all_ci_upper[pat])
      name = 'John Doe' if icu_pat.loc[pat, 'GENDER_M'] else 'Jane Doe'
      dob = (icu_pat.loc[pat, 'INTIME'] - pd.to_timedelta(icu_pat.loc[pat, 'AGE'], unit='d')).strftime('%Y-%m-%d')
      patient_id = icu_pat.loc[pat, 'SUBJECT_ID']
      recent_admissions = icu_pat.loc[pat, 'NUM_RECENT_ADMISSIONS']

      mu_charts = mu[-num_charts:]
      scores_charts_pat = all[pat, -num_charts:] * mu_charts
      relevance_order_charts_pat = [11, 1, 7, 10] #np.argsort(-scores_charts_pat)[:4]
      charts_pat = charts[pat, relevance_order_charts_pat, :]
      chart_labels_pat = chart_labels[relevance_order_charts_pat]      
      activations_pat = charts_activations[pat] * np.expand_dims(mu_charts, -1)
      activations_pat = activations_pat[relevance_order_charts_pat, :]
      activations_pat = activations_pat / np.max(np.abs(activations_pat)) -0.1
      
      df_diag = pd.DataFrame({'DIAGNOSES': diag[pat], 'WEIGHT': weights_diag[pat]})
      df_diag['COEFF'] = df_diag['DIAGNOSES'].apply(lambda x: mean_coeff_diag[x])
      df_diag['COEFF'] = df_diag['COEFF']*df_diag['WEIGHT']
      df_diag['COEFF'] = df_diag['COEFF']/np.max(np.abs(df_diag['COEFF']))
      df_diag = df_diag.loc[(df_diag['DIAGNOSES']>0) & ~(df_diag['DIAGNOSES']==934)] #pad and other
      df_diag.sort_values(by='COEFF', ascending=False, inplace=True)
      df_diag = df_diag.head(7)
      df_diag['DIAGNOSES'] = df_diag['DIAGNOSES'].replace(dict_diagnoses).replace(d_icd_diagnoses)
      df_diag['DIAGNOSES'] = df_diag['DIAGNOSES'].apply(lambda x: x[:28] + '..' if len(x)>30 else x)

      df_proc = pd.DataFrame({'PROCEDURES': proc[pat], 'WEIGHT': weights_proc[pat]})
      df_proc['COEFF'] = df_proc['PROCEDURES'].apply(lambda x: mean_coeff_proc[x])
      df_proc['COEFF'] = df_proc['COEFF']*df_proc['WEIGHT']
      df_proc['COEFF'] = df_proc['COEFF']/np.max(np.abs(df_proc['COEFF']))
      df_proc = df_proc.loc[(df_proc['PROCEDURES']>0) & ~(df_proc['PROCEDURES']==312)] #pad and other
      df_proc.sort_values(by='COEFF', ascending=False, inplace=True)
      df_proc = df_proc.head(7)
      df_proc['PROCEDURES'] = df_proc['PROCEDURES'].replace(dict_procedures).replace(d_icd_procedures)
      df_proc['PROCEDURES'] = df_proc['PROCEDURES'].apply(lambda x: x[:28] + '..' if len(x)>30 else x)
      
      df_pres = pd.DataFrame({'PRESCRIPTIONS': pres[pat], 'WEIGHT': weights_pres[pat]})
      df_pres['COEFF'] = df_pres['PRESCRIPTIONS'].apply(lambda x: mean_coeff_pres[x])
      df_pres['COEFF'] = df_pres['COEFF']*df_pres['WEIGHT']
      df_pres['COEFF'] = df_pres['COEFF']/np.max(np.abs(df_pres['COEFF']))
      df_pres = df_pres.loc[(df_pres['PRESCRIPTIONS']>0) & ~(df_pres['PRESCRIPTIONS']==423)] #pad and other
      df_pres.sort_values(by='COEFF', ascending=False, inplace=True)
      df_pres = df_pres.head(7)
      df_pres['PRESCRIPTIONS'] = df_pres['PRESCRIPTIONS'].replace(dict_prescriptions)
      df_pres['PRESCRIPTIONS'] = df_pres['PRESCRIPTIONS'].apply(lambda x: x[:28] + '..' if len(x)>30 else x)
      
      if (len(df_diag) == 7) and (len(df_proc) == 7) and (len(df_pres) == 7):
      
        fig = plt.figure()
        plt.rcParams.update({'font.size': 6, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5})
        outer = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.1, height_ratios=[1, 2, 2])
        
        ################ Top figure ###########################
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        ax = plt.Subplot(fig, inner[0])
        ax.set_aspect('equal')  
        ax.set_xticks([])
        ax.set_yticks([])    
        fig.add_subplot(ax)
        
        p = avg_pred/100
        pie_center = (3,-0.1)
        patches = plt.pie([p,1-p], center=pie_center, startangle=90, counterclock=False, colors=[cm.coolwarm(p), 'white'], wedgeprops={'edgecolor':'k', 'linewidth':0.5})
        patches[0][1].set_alpha(0)
        patches = plt.pie([p,1-p], center=pie_center, startangle=90, counterclock=False, colors=['white', 'white'], wedgeprops={'edgecolor':'k', 'linewidth':0.5}, radius=0.8)
        patches[0][1].set_alpha(0)
        circle = plt.Circle(pie_center, 0.78, facecolor='white', linewidth=0.5, clip_on=False)
        ax.add_artist(circle)
        plt.text(pie_center[0], pie_center[1], str(avg_pred)+'%', fontsize=12, horizontalalignment='center', verticalalignment='center')
        plt.text(-9, 1.3, 'Risk of ICU Readmission Within 30 Days', fontsize=11, weight='semibold')
        plt.text(-9, 0.6, 'Name: ' + name, fontsize=6)
        plt.text(-9, 0., 'DOB: ' + dob, fontsize=6)
        plt.text(-9, -0.6, 'Patient ID: ' + str(patient_id), fontsize=6)
        plt.text(-9, -1.2, 'Num. ICU Admissions Last Year: ' + str(recent_admissions), fontsize=6)
        plt.text(pie_center[0]+0.8, pie_center[1]-1.1, 'CI: ' + str(ci_lower) + '% to ' + str(ci_upper) + '%', verticalalignment='bottom', style='italic', fontsize=6)
        ax.set_xlim([-1.25, 1.25])
        ax.set_ylim([-1.25, 1.25])  
        #######################################################
        
        ################ Charts ###########################
        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], wspace=0.2, hspace=0.8)
        
        x_vals = np.linspace(0, -47, 1000)
        for i in range(4):
          ax = plt.Subplot(fig, inner[i])
          fy = interp1d(range(0, -48, -1), charts_pat[i, :], kind='cubic')
          fc = interp1d(range(0, -48, -1), medfilt(activations_pat[i, :], kernel_size=5), kind='cubic')
          ax.scatter(x_vals, fy(x_vals), c=cm.coolwarm(fc(x_vals)), s=2, edgecolor='none')
          if (i==2) or (i==3):
            ax.xaxis.set_ticklabels([])
            ax.xaxis.set_ticks_position('top')
          else:
            ax.set_xlabel('Time [h]')
          ax.set_ylabel(chart_labels_pat[i])
          fig.add_subplot(ax)

        #######################################################

        ################ Tables ###########################
        inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.1, hspace=0.1)

        if (len(df_diag) > 0):
          ax = plt.Subplot(fig, inner[0])
          ax.axis('off')
          colors = 0.5*df_diag[['COEFF']].values+0.4
          colors = [[rgb2hex(cm.coolwarm(float(c)))] for c in colors]
          tab = ax.table(cellText=df_diag[['DIAGNOSES']].values, colLabels=df_diag[['DIAGNOSES']].columns, loc='center', cellLoc='left', fontsize=6, cellColours=colors)
          for key, cell in tab.get_celld().items():
            cell.set_linewidth(0.5)
          tab.auto_set_font_size(False)
          tab.set_fontsize(6)
          ax.set_xticks([])
          ax.set_yticks([])
          fig.add_subplot(ax)

        if (len(df_proc) > 0):
          ax = plt.Subplot(fig, inner[1])
          ax.axis('off')
          colors = 0.5*df_proc[['COEFF']].values+0.4
          colors = [[rgb2hex(cm.coolwarm(float(c)))] for c in colors]
          tab = ax.table(cellText=df_proc[['PROCEDURES']].values, colLabels=df_proc[['PROCEDURES']].columns, loc='center', cellLoc='left', fontsize=6, cellColours=colors)
          for key, cell in tab.get_celld().items():
            cell.set_linewidth(0.5)
          tab.auto_set_font_size(False)
          tab.set_fontsize(6)
          ax.set_xticks([])
          ax.set_yticks([])
          fig.add_subplot(ax)

        if (len(df_pres) > 0):
          df_pres = df_pres.rename(columns={'PRESCRIPTIONS':'MEDICATIONS'})
          ax = plt.Subplot(fig, inner[2])
          ax.axis('off')
          colors = 0.5*df_pres[['COEFF']].values+0.4
          colors = [[rgb2hex(cm.coolwarm(float(c)))] for c in colors]
          tab = ax.table(cellText=df_pres[['MEDICATIONS']].values, colLabels=df_pres[['MEDICATIONS']].columns, loc='center', cellLoc='left', fontsize=6, cellColours=colors)
          for key, cell in tab.get_celld().items():
            cell.set_linewidth(0.5)
          tab.auto_set_font_size(False)
          tab.set_fontsize(6)
          ax.set_xticks([])
          ax.set_yticks([])
          fig.add_subplot(ax)
            
        #######################################################
        
        # plt.show()
        print('{} {}'.format(icu_pat.loc[pat, 'SUBJECT_ID'], icu_pat.loc[pat, 'AGE']))
        plt.savefig('../reports/' + str(avg_pred) + '_' + str(patient_id) + '.pdf', bbox_inches='tight')
        plt.close()
        # bp()
  print('-----------------------------------------')
  
  
  
  
  