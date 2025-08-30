""" create config file and subject specific output directories for movienf experiment
"""
import subprocess
import toml
import os
import time
import sys
import datetime
import pandas as pd
import numpy as np


date_today = str(datetime.date.today()).replace('-','')

# load config file
project = 'movienf'

# get root directory
file_path = os.path.realpath(__file__)
root_dir = os.path.dirname(file_path)
root_dir_analysis = root_dir
root_dir_task = root_dir
reference_dir = os.path.join(root_dir, 'referenceDir')
dicom_dir = os.path.join(root_dir, 'dicomDir')
conf_init_path = os.path.join(root_dir, 'conf','{}_init.toml'.format(project))
conf_output_path = conf_init_path.replace('_init','')
conf = toml.load(conf_init_path)

# collect subject_id and session number in terminal
subject_id = input("Enter subject ID: ")
session = input("Enter session number: ")
run = input("Enter run number: ")

# update conf file
conf['subjectNum'] = subject_id
conf['subjectDay'] = session
conf['runNum'] = run
conf['date'] = date_today

conf['dicomNamePattern'] = "001_{RUN:06}_{TR:06}.dcm"

# TRs and timing
conf['movie_trs'] = 1023
conf['disdaqs'] = 20 # number of disdaqs to collect before starting the experiment
conf['hrf_delay'] = 2 # delay in seconds for the HRF
conf['hrf_delay_min'] = 1 # minimum delay in seconds for the HRF
conf['hrf_delay_max'] = 10 # maximum delay in seconds for the HRF

# add number of stations and total feedback duration
conf['numStations'] = 8 # number of stations to run in parallel
conf['feedbackWaitTime'] = 10 # total feedback duration in seconds
conf['feedbackDisplayTime'] = 10 # time to display feedback in seconds
conf['feedbackContinueTime'] = 2 # time to continue feedback in seconds
conf['feedbackTimeTotal'] = conf['feedbackWaitTime'] + conf['feedbackDisplayTime'] + conf['feedbackContinueTime']

# total TRs
conf['total_trs'] = conf['movie_trs'] + conf['disdaqs'] + conf['hrf_delay_max'] + conf['numStations']*conf['feedbackTimeTotal'] # total number of TRs to collect
# total trs = 1023 + 20 + 10 + 8*(10+10+2) = 1229

def get_data_dirs(root_dir, exp_info, task='movienf'):
    # get skyport directories
    subject_dir = os.path.join(root_dir, 'data', 'sub-{}'.format(exp_info['Subject']))
    session_dir = os.path.join(subject_dir, 'ses-{}'.format(exp_info['Session']))
    run_dir = os.path.join(session_dir, 'task-{}_run-{}'.format(task, exp_info['Run']))
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    return subject_dir, session_dir, run_dir

# add data directories
# analysis machine
_exp_info = {'Subject': subject_id, 'Session': session, 'Run': run}
subject_dir_analysis, session_dir_analysis, nf_run_dir_analysis = get_data_dirs(root_dir_analysis, _exp_info, task='nf')
subject_dir_analysis, session_dir_analysis, reference_run_dir_analysis = get_data_dirs(root_dir_analysis, _exp_info, task='reference')
# task machine
subject_dir_task, ssession_dir_task, nf_run_dir_task = get_data_dirs(root_dir_task, _exp_info, task='nf')
subject_dir_task, session_dir_task, reference_run_dir_task = get_data_dirs(root_dir_task, _exp_info, task='reference')

# make run level directory if it doesn't exist
if not os.path.exists(nf_run_dir_analysis):
    os.makedirs(nf_run_dir_analysis, exist_ok=True)

# protocol names for dcm2niix conversion
conf['reference_protocol'] = 'BOLD_RL_movie_nf_reference'
conf['nfprotocol'] = 'BOLD_RL_movie_nf_trainspotting'

# directories 
conf['dicomDirAnalysis'] = dicom_dir
conf['referenceDir'] = reference_dir
conf['runDirAnalysis'] = nf_run_dir_analysis
conf['sessionDirAnalysis'] = session_dir_analysis
conf['rootDirAnalysis'] = root_dir_analysis
conf['rootDirTask'] = root_dir_task
conf['runDirTask'] = nf_run_dir_task


# reference file paths
conf['refseries'] = os.path.join(reference_dir, 'sub-{}_ses-{}_refseries.nii.gz'.format(subject_id, session))
conf['refvol']= os.path.join(reference_dir, 'sub-{}_ses-{}_refvol.nii.gz'.format(subject_id, session))
conf['refvol_brain'] = os.path.join(reference_dir, 'sub-{}_ses-{}_refvol_brain.nii.gz'.format(subject_id, session))
conf['refvol_brain_mask'] = os.path.join(reference_dir, 'sub-{}_ses-{}_refvol_brain_mask.nii.gz'.format(subject_id, session))
mask_fname = 'dlpfc_weights_mask'
conf['standard_nf_mask'] = os.path.join(root_dir, 'clf', '{}.nii.gz'.format(mask_fname))
conf['subject_nf_mask'] = os.path.join(reference_dir, 'sub-{}_ses-{}_{}.nii.gz'.format(subject_id, session, mask_fname))


# write to conf file
#--------------------------------------------------------------------------
# save a copy of the config file with a generic name for access by the display computer to initialize the experiment
conf_output_path_general = os.path.join(root_dir_analysis, '{project}.toml'.format(project=project))    
# save a copy to the specific session
conf_output_path_session = os.path.join(nf_run_dir_analysis, '{project}.toml'.format(project=project))
conf_output_path_local = os.path.join(root_dir, 'conf','{}.toml'.format(project))
with open(conf_output_path_general, 'w') as f:
    toml.dump(conf, f)
with open(conf_output_path_session, 'w') as f:
    toml.dump(conf, f)
with open(conf_output_path_local, 'w') as f:
    toml.dump(conf, f)

# run data analyzer
#--------------------------------------------------------------------------
# dont need cryptography
# os.system("export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1")
# os.system("export TMPDIR=/home/brain/tmp")
subprocess.call("WEB_IP=localhost", shell=True)
# subprocess.call("conda activate rtcloud")
# subprocess.call("cd rt-cloud", shell=True)
subprocess.call("bash scripts/data_analyser.sh -p {} --test".format(project), shell=True)