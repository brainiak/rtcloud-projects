import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from subprocess import call
from pathlib import Path
from datetime import datetime, date
from nilearn.image import get_data
from nilearn.signal import clean
from nilearn.glm.first_level.design_matrix import _cosine_drift

from utils import *

print('************************************************')
cwd = os.getcwd()
print(f"rt-cloud folder: {cwd}")
sys.path.append(cwd)
from rtCommon.utils import loadConfigFile

projectDir = os.path.dirname(os.path.realpath(__file__))
print(f'project folder: {projectDir}')

# obtain the full path for the configuration toml file
# if the toml variables have been changed in the web interface 
# then use those altered variables instead (note: does not overwrite the toml)
defaultConfig = os.path.join(cwd, f'conf/{Path(__file__).stem}.toml')
argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                        help='experiment config file (.json or .toml)')
args = argParser.parse_args(None)
cfg = loadConfigFile(args.config)

#collect and format the subject number and session
sub = cfg.subjectNum
subj = f'sub-RP{sub:03d}'

ses = cfg.subjectDay
sess = f'ses-{ses}'

tnt_n_trs = int(cfg.tnt_n_trs)

#this is the output from the processing, where i can also put the files i make here
outRoot = cfg.outRoot
outDir = f'{outRoot}/{subj}/{sess}'
print(f'output folder: {outDir}')
print('************************************************')


"""-----------------------------------------------------------------------------
this part creates the events file from the psychopy output
-----------------------------------------------------------------------------"""
tnt_files = [i for i in os.listdir(outDir) if '.csv' in i and 'task-tnt' in i and 'confounds' not in i and 'events' not in i]

for f, file in enumerate(tnt_files):
    print(f'Creating events file {int(f+1)}')
    df = pd.read_csv(f'{outDir}/{file}')
    df = df.drop([0,df.index[-1]])
    df['subject'] = sub
    df['session'] = ses
    df.session = df.session.astype(int)
    
    zero_time = df.loc[1,'fixation_2.started']
    
    df['onset'] = df['cue_text.started'] - zero_time
    df['duration'] = 3
    df['response'] = np.nan
    df['response_time'] = np.nan

    df.repetition = df.repetition.astype(int)
    df.pair_num = df.pair_num.astype(int)
    
    for i in df.index:
    
        if df.loc[i,'tnt_feedback_resp.keys'] != 'None':
            df.loc[i,'response'] = int(df.loc[i,'tnt_feedback_resp.keys'])
            df.loc[i,'response_time'] = df.loc[i,'tnt_feedback_resp.rt']
        
        elif df.loc[i,'iti_feedback_resp.keys'] != 'None':
            df.loc[i,'response'] = int(df.loc[i,'iti_feedback_resp.keys'])
            df.loc[i,'response_time'] = df.loc[i,'iti_feedback_resp.rt'] + 1

    df = df.rename(columns={'condition':'trial_type'})

    df = df[['onset','duration','trial_type','response','response_time','repetition','cue','target','pair_num','subject','session']]

    df['target_status'] = ''
    for i in df.index:
        if np.isnan(df.loc[i,'response']):
            if df.loc[i,'trial_type'] == 'think':
                df.loc[i,'target_status'] = 'retrieval'
            elif df.loc[i,'trial_type'] == 'no-think':
                df.loc[i,'target_status'] = 'suppression'

        elif df.loc[i,'trial_type'] == 'think' and df.loc[i,'response'] == 1:
            df.loc[i,'target_status'] = 'retrieval'
        
        elif df.loc[i,'trial_type'] == 'think' and df.loc[i,'response'] == 2:
            df.loc[i,'target_status'] = 'no_retrieval'
        
        elif df.loc[i,'trial_type'] == 'no-think' and df.loc[i,'response'] == 1:
            df.loc[i,'target_status'] = 'intrusion'
        
        elif df.loc[i,'trial_type'] == 'no-think' and df.loc[i,'response'] == 2:
            df.loc[i,'target_status'] = 'suppression'

    out_file = file.split('_2023')[0]

    df.trial_type = df.trial_type.apply(lambda x: x.replace('-','_'))

    df.to_csv(f'{outDir}/{out_file}_events.csv',index=False)

"""-----------------------------------------------------------------------------
this part runs the glms to create the baseline estimate in no-think connectivity
between the MFG and bilateral hippocampus
-----------------------------------------------------------------------------"""
hc_mask = get_data(f'{outDir}/{subj}_{sess}_HC_mask.nii.gz')
mfg_mask = get_data(f'{outDir}/{subj}_{sess}_MFG_mask.nii.gz')
brain_mask = get_data(f'{outDir}/{subj}_{sess}_refvol_brain_mask.nii.gz')

tnt_runs = [1,2,3,4]

#load in the images
imgs = [nib.load(f'{outDir}/{subj}_{sess}_task-tnt_run-{run}_desc-bold_mc_sm.nii.gz') for run in tnt_runs]

#get the events
events = [pd.read_csv(f'{outDir}/{subj}_{sess}_task-tnt_run-{run}_events.csv') for run in tnt_runs]

#load in the confounds
confounds = [pd.read_csv(f'{outDir}/{subj}_{sess}_task-tnt_run-{run}_confounds.csv') for run in tnt_runs]
pad_print(f'confounds found: {confounds[0].columns}; (high-pass regs added later)')
ests = []
'''run the glm'''
for r, run in enumerate(tnt_runs):
    print(f'Running GLM for run {run}')
    #load in the images
    img = imgs[r]
    #get the events and add trial_type column
    e = events[r]
    #load in the confounds
    c = confounds[r]

    '''load in wb, mfg, and hc'''
    img_data = img.get_fdata()
    wb_data = fast_apply_mask(img_data,brain_mask)
    mfg_ts = fast_apply_mask(img_data,mfg_mask).mean(1)
    hc_ts = fast_apply_mask(img_data,hc_mask)
    
    '''scale mfg and hc by wb'''
    mfg_ts_scaled = ref_mean_scaling(mfg_ts,wb_data,(0,1))
    hc_ts_scaled = ref_mean_scaling(hc_ts,wb_data,(0,1))

    hp_regs = _cosine_drift((1/128), np.linspace(0*2, (tnt_n_trs-1+0)*2, tnt_n_trs))
    c = np.hstack((c.values,hp_regs))

    '''clean both, this could be more elegant but it works here'''
    seed_ts = clean(mfg_ts_scaled, detrend=False, standardize=False, confounds=c, standardize_confounds=True, 
                                   filter=None, high_pass=None, t_r=2.0)

    Y = clean(hc_ts_scaled, detrend=False, standardize=False, confounds=c, standardize_confounds=True, 
                            filter=None, high_pass=None, t_r=2.0)
        
    for i in range(e.shape[0]):
        lss_df = e.copy()
        if lss_df.loc[i,'trial_type'] == 'no_think':
            lss_df.loc[i,'trial_type'] = 'target'
            
            est, dm = fast_glm(Y, lss_df, 'target_ppi', confounds=None, seed_ts=seed_ts, slice_time_ref=0.,
                      t_r=2, drift_model=None, high_pass=None, noise_model='ols')
            ests.append(est)
baseline_scores = np.array(ests)
score_zero = np.mean(baseline_scores)
score_std = np.std(baseline_scores,ddof=1)
np.savetxt(f'{outRoot}/{subj}/baseline_score.txt',[score_zero,score_std])

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'score_zero is {np.round(score_zero,4)}, score_std is {np.round(score_std,4)}')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
