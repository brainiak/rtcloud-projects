"""-----------------------------------------------------------------------------
Imports
-----------------------------------------------------------------------------"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import argparse
import json
import tempfile
import time
import nibabel as nib
import pandas as pd
import numpy as np

from subprocess import call
from pathlib import Path
from datetime import datetime, date
from scipy.stats import zscore
from nilearn.image import get_data, new_img_like
from nilearn.signal import clean
from nilearn.glm.first_level.design_matrix import _cosine_drift

from utils import *

"""-----------------------------------------------------------------------------
Load project configuration and initialize paths & variables 
-----------------------------------------------------------------------------"""
print('************************************************')
cwd = os.getcwd()
print(f"rt-cloud folder: {cwd}")
sys.path.append(cwd)

from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface
from rtCommon.imageHandling import readRetryDicomFromDataInterface, convertDicomImgToNifti, saveAsNiftiImage
from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsRun import BidsRun

'''project directory'''
projectDir = os.path.dirname(os.path.realpath(__file__)) #'.../rt-cloud/projects/project_name'
print(f"project folder: {projectDir}")

'''load the config file'''
defaultConfig = os.path.join(projectDir, f'conf/{Path(__file__).stem}.toml')
argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                        help='experiment config file (.json or .toml)')
args = argParser.parse_args(None)
cfg = loadConfigFile(args.config)

'''collect and format the subject and session number'''
sub = cfg.subjectNum
subj = f'sub-RP{sub:03d}'

ses = cfg.subjectDay
sess = f'ses-{ses}'

'''find the dicom folder, which needs the datestring in the format of YYYYMMDD'''
dicomRoot = cfg.dicomRoot
if dicomRoot == 'None':
    dicomRoot = f'{projectDir}/dicomDir'

dateString = cfg.dateString
if dateString == "default":
    today = date.today()
    dateString = today.strftime('%Y%m%d')
dicomPath = f'{dicomRoot}/{dateString}.sub_RP{sub:03d}.sub_RP{sub:03d}'
print(f'dicom folder: {dicomPath}')

'''this is the output from the processing, where i can also put the files i make here'''
outRoot = cfg.outRoot
if outRoot == 'None':
    outRoot = f'{projectDir}/outDir'
outDir = f'{outRoot}/{subj}/{sess}'
if not os.path.exists(f"{outDir}"):
    os.makedirs(f"{outDir}")
print(f'output folder: {outDir}')

'''specify directory for temporary files'''
tmpPath = f'{outRoot}/tmp'
if os.path.exists(tmpPath):
    os.system(f'rm -r {tmpPath}')
    os.makedirs(tmpPath)
else:
    os.makedirs(tmpPath)
print(f"temp folder: {tmpPath}")

print('************************************************')
print(f"\n----Starting project: {cfg.title}----\n")

#starting task, default varies by session
task = cfg.task
if ses > 1:
    task = 'feedback'
elif task == 'default':
    task = 'tnt'
#starting run number
start_run = int(cfg.runNum[0])

#number of TRs/runs in each task
feedback_n_trs = int(cfg.feedback_n_trs)
feedback_n_runs = int(cfg.feedback_n_runs)

tnt_n_trs = int(cfg.tnt_n_trs)
tnt_n_runs = int(cfg.tnt_n_runs)

#this is stable across all runs of the feedback, high-pass cosine filters
hp_regs = _cosine_drift((1/128), np.linspace(0*2, (feedback_n_trs-1+0)*2, feedback_n_trs))

#what is the number of scans to account for when looking for real-time dicoms
#real-time should be started after the auto-scout, before the refseries
num_runs_before_rt_flag = cfg.num_runs_before_rt_flag
if num_runs_before_rt_flag == 'default':
    if ses == 1:
        if task == 'tnt':
            num_runs_before_rt = 1
        elif task == 'feedback':
            num_runs_before_rt = 6    
    if ses == 2:
        num_runs_before_rt = 2
else:
    num_runs_before_rt = int(num_runs_before_rt_flag)

#flag for sending results to analysisListner
send_results = cfg.send_results

#flag for if we are running a sham participant or not
sham = cfg.sham

# Initialize the remote procedure call (RPC) for the data_analyser
# (aka projectInferface). This will give us a dataInterface for retrieving 
# files, a subjectInterface for giving feedback, a webInterface
# for updating what is displayed on the experimenter's webpage,
# and enable BIDS functionality
clientInterfaces = ClientInterface(rpyc_timeout=999999)
webInterface  = clientInterfaces.webInterface
bidsInterface = clientInterfaces.bidsInterface
subjInterface = clientInterfaces.subjInterface
subjInterface.subjectRemote = True

"""====================REAL-TIME ANALYSIS GOES BELOW====================
Use the below section to program whatever real-time analysis that you want  
performed on your scanning data. In this example, for each TR,
we transform the DICOM data into a Nifti file and then apply motion correction 
and spatial smoothing. We then mask voxels and save the activations 
for later training of the multivoxel classifier.
===================================================================="""
# clear existing web browser plots if there are any
try:
    webInterface.clearAllPlots()
except:
    pass

"""-----------------------------------------------------------------------------
Load in things needed for the real-time analysis, here ROI masks and TR labels
-----------------------------------------------------------------------------"""
'''refvol and masks'''
refvol = f'{outDir}/{subj}_{sess}_refvol.nii.gz'
brain_mask = get_data(f'{outDir}/{subj}_{sess}_refvol_brain_mask.nii.gz')

MFG_mask = get_data(f'{outDir}/{subj}_{sess}_MFG_mask.nii.gz')
HC_mask = get_data(f'{outDir}/{subj}_{sess}_HC_mask.nii.gz')

'''TR events and labels'''
timing = {}
for i in range(start_run,feedback_n_runs+1):
    try:
        timing[i] = {'events': pd.read_csv(f'{outDir}/{subj}_{sess}_task-feedback_run-{i}_events.csv'),
                     'labels': pd.read_csv(f'{outDir}/{subj}_{sess}_task-feedback_run-{i}_labels.csv').set_index(['TR'])}
    except:
        print(f'@@@loading events/labels for {subj}_{sess}_run-{i} failed@@@')

"""-----------------------------------------------------------------------------
Think/no-think loop starts here. The goal is just to do real-time pre-processing
so that we can quickly do GLM estimation when all runs are done
-----------------------------------------------------------------------------"""
if task == 'tnt':

    for run in range(start_run,tnt_n_runs+1):
        timer = tr_timer()
        
        #plus something here to account for the shift in things from the auto-scout
        dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'RUN', run+num_runs_before_rt)
        streamId = bidsInterface.initDicomBidsStream(dicomPath,dicomScanNamePattern,
                                                    cfg.minExpectedDicomSize, 
                                                    anonymize=False,
                                                    **{'subject':cfg.subjectNum,
                                                    'run':run,
                                                    'task':'tnt'})
        for TR in range(1,tnt_n_trs+1):
            pad_print(f'--- Task TNT | Run {run} | TR {TR} ---')
            bidsIncremental = bidsInterface.getIncremental(streamId,volIdx=TR,
                                            timeout=999999,demoStep=cfg.demoStep)
            
            timer.tr_diff(TR)#calculate the time difference between TRs
            nib.save(bidsIncremental.image, f'{tmpPath}/temp.nii')#save the image

            #run motion correction and smoothing
            motion_correct(in_=f'{tmpPath}/temp.nii', reffile=refvol, out=f'{tmpPath}/temp_aligned')
            smooth(fwhm=5, in_=f'{tmpPath}/temp_aligned', out=f'{tmpPath}/temp_aligned_smoothed')

            if TR == 1:
                img = get_data(f'{tmpPath}/temp_aligned_smoothed.nii.gz')[:,:,:,np.newaxis] #we want the full 4D array
                mc_params = np.loadtxt(f'{tmpPath}/temp_aligned.par') #motion paramss

            else:
                #otherwise we just concat them
                img = np.concatenate((img, get_data(f'{tmpPath}/temp_aligned_smoothed.nii.gz')[:,:,:,np.newaxis]), -1)
                mc_params = np.vstack((mc_params, np.loadtxt(f'{tmpPath}/temp_aligned.par')))

        out_img = new_img_like(bidsIncremental.image, img, copy_header=True)
        nib.save(out_img, f'{outDir}/{subj}_{sess}_task-tnt_run-{run}_desc-bold_mc_sm.nii.gz')

        confounds = pd.DataFrame(mc_params, columns=[f'mc-{i}' for i in range(6)])
        confounds.to_csv(f'{outDir}/{subj}_{sess}_task-tnt_run-{run}_confounds.csv',index=False)
        
        bidsInterface.closeStream(streamId)
        print(f"-----------------------------------------------------------------------\n"
        f"END OF TNT RUN {run}!\n"
        "-----------------------------------------------------------------------")
    
    #we have to quit out in order to calculate the baseline on day 1
    print("-----------------------------------------------------------------------\n"
    "TNT EXPERIMENT COMPLETE\nRUN FINALIZE SCRIPT\n"
    "-----------------------------------------------------------------------")
    sys.exit(0)


"""-----------------------------------------------------------------------------
feedback loop starts here
-----------------------------------------------------------------------------"""
for run in range(start_run,feedback_n_runs+1):
    total_score = 0
    timer = tr_timer()

    '''make the run outDir'''
    run_outDir = f'{outDir}/task-feedback_run-{run}'
    os.makedirs(run_outDir,exist_ok=True)

    '''check to see if its empty or not'''
    if len([i for i in os.listdir(run_outDir)]) > 3:
        print('************************************************')
        print('LOOKS LIKE YOU ARE RERUNNING A RUN THAT ALREADY HAS DATA')
        print('WAITING 5 SECONDS AND THEN OVERWRITING EXISTING OUTPUT')
        print('************************************************')
        time.sleep(5)
        os.system(f'rm -r {run_outDir}')
        os.makedirs(run_outDir,exist_ok=True)

    '''lets get the run events now and add the feedback, since we only have to do that once'''
    run_events = timing[run]['events'].copy()
    for i in range(run_events.shape[0]):
        onset, duration, post_induction_dur, calculation_dur, feedback_dur, trial_num = run_events.loc[i,['onset','duration','post_induction_dur','calculation_dur','feedback_dur','trial_num']]

        new_row = pd.DataFrame({'onset': (onset+duration+post_induction_dur+calculation_dur), 'duration': feedback_dur, 'trial_type':'feedback', 'trial_num':trial_num},index=[0])
        run_events = pd.concat((run_events,new_row))

    run_events = run_events.sort_values(by='onset').reset_index(drop=True)
    '''load in the baseline scores'''
    scores = load_baseline(f'{outRoot}/{subj}/baseline_score.txt')

    # prep stream of DICOMS -> BIDS
    if num_runs_before_rt_flag == 'default' and ses == 1 and task == 'tnt':
        add_n_runs = num_runs_before_rt + tnt_n_runs + 1
    else:
        add_n_runs = num_runs_before_rt

    dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'RUN', run+add_n_runs)#plus something here to account for the shift in things from the auto-scout
    streamId = bidsInterface.initDicomBidsStream(dicomPath,dicomScanNamePattern,
                                                cfg.minExpectedDicomSize, 
                                                anonymize=False,
                                                **{'subject':cfg.subjectNum,
                                                'run':run,
                                                'task':'feedback'})

    # reset the first x-axis plot location for Data Plot
    point_idx=0

    #need to init this var here
    calc_dur = 0

    if sham:
        pad_print('WARNING: RUNNING SHAM NEUROFEEDBACK')
    else:
        pad_print('Delivering activate neurofeedback')
    """-----------------------------------------------------------------------------
    real-time processing starts in this loop
    -----------------------------------------------------------------------------"""
    dms_out = []
    events_out = [] 
    # TR will be our run-specific TR counter
    for TR in range(1,feedback_n_trs+1):
        
        #find what is supposed to be happening this TR
        tr_label = timing[run]['labels'].loc[TR,'label']
        pad_print(f'--- Run {run} | TR {TR} | Label {tr_label} ---')
        
        '''we check if instead of loading a new TR, we should instead do a GLM calculation on previous data'''
        if tr_label == 'calculation':
            #start the timer
            timer.calc_diff('start')

            #get the trial number
            trial_num = int(timing[run]['labels'].loc[TR,'trial_num'])
            pad_print(f'Running GLM for trial {trial_num}')  
            
            #need to adjust the timing of the previous trial feedback PERMANETNLY            
            if calc_dur:
                feedback_time_adjust = 2-calc_dur
                index_to_change = run_events.loc[run_events.trial_type == 'feedback'].loc[run_events.trial_num == int(trial_num - 1),'trial_type'].index
                run_events.loc[index_to_change,'onset'] = run_events.loc[index_to_change,'onset'] - feedback_time_adjust
                run_events.loc[index_to_change,'duration'] = run_events.loc[index_to_change,'duration'] + feedback_time_adjust

            #copy the events so we can change trial_type label to target
            events = run_events.copy()
            events = events[events.trial_num <= trial_num]
            
            index_to_change = events.loc[events.trial_type == 'induction'].loc[events.trial_num == trial_num,'trial_type'].index
            events.loc[index_to_change,'trial_type'] = 'target'

            #if its the first trial get rid of the feedback
            if trial_num == 1:
                events = events.drop(events.loc[events.trial_type == 'feedback'].loc[events.trial_num == 1,'trial_type'].index)
            
            #get the confounds
            confounds = np.hstack((mc_params,hp_regs[:mc_params.shape[0]]))
            pad_print(f'{int(confounds.shape[1])} found (15 expected)')
            #scale the data based on the first 60s
            hc_ts_scaled = ref_mean_scaling(hc_ts, scaling_ref, (0,1))
            mfg_ts_scaled = ref_mean_scaling(mfg_ts, scaling_ref, (0,1))

            #clean the seed
            seed_ts = clean(mfg_ts_scaled, detrend=False, standardize=False, confounds=confounds, standardize_confounds=True, 
                                           filter=None, high_pass=None, t_r=2.0)

            #clean the data
            Y = clean(hc_ts_scaled, detrend=False, standardize=False, confounds=confounds, standardize_confounds=True, 
                                    filter=None, high_pass=None, t_r=2.0)
            
            #run GLM
            est, dm = fast_glm(Y, events, 'target_ppi', confounds=None, seed_ts=seed_ts, slice_time_ref=0.,
                                                      t_r=2, drift_model=None, high_pass=None, noise_model='ols')

            #score calculation goes here, the -1 is to flip the sign, and min-max to -1,1
            if sham:
                score = np.clip( ((est - scores.score_zero) / scores.score_std), -1, 1)
            else:
                score = np.clip( ((est - scores.score_zero) / scores.score_std) * -1, -1, 1)

            #figure out how long it took
            calc_dur = timer.calc_diff('stop')
            pad_print(f'GLM time: {np.round(calc_dur,4)}')

            if send_results:
                subjInterface.setResultDict(name=f'{subj}/{sess}/run-{run}/trial-{int(trial_num)}',
                                    values={'score': str(score),
                                            'est': str(est),
                                            'trial_num':str(int(trial_num)),
                                            'sham': str(sham),
                                            'calc_dur': str(calc_dur)})


            #update plot on web interface
            point_idx+=1
            webInterface.plotDataPoint(run, point_idx, score)

            dms_out.append(dm)
            events_out.append(events)
            #add score to total
            total_score += score

        '''the rest happens regardless, but only after GLM if needed'''
        bidsIncremental = bidsInterface.getIncremental(streamId,volIdx=TR,
                                        timeout=999999,demoStep=cfg.demoStep)
        
        timer.tr_diff(TR)#calculate the time difference between TRs
        nib.save(bidsIncremental.image, f'{tmpPath}/temp.nii')#save the image

        #run motion correction and smoothing
        motion_correct(in_=f'{tmpPath}/temp.nii', reffile=refvol, out=f'{tmpPath}/temp_aligned')
        smooth(fwhm=5, in_=f'{tmpPath}/temp_aligned', out=f'{tmpPath}/temp_aligned_smoothed')

        # Load nifti data as img variable
        img = get_data(f'{tmpPath}/temp_aligned_smoothed.nii.gz')[:,:,:,np.newaxis] #we want the full 4D array
        if TR == 1:
            out_img = np.copy(img)
            scores.check()
            wb_data = fast_apply_mask(img, brain_mask) #the brain masked data
            mfg_ts = fast_apply_mask(img, MFG_mask).mean() #the mfg time series
            hc_ts = fast_apply_mask(img, HC_mask) #the HC time series
            mc_params = np.loadtxt(f'{tmpPath}/temp_aligned.par') #motion paramss

        else:
            #otherwise we just concat them
            out_img = np.concatenate((out_img, img), -1)
            wb_data_this_tr = fast_apply_mask(img,brain_mask)
            wb_data = np.vstack((wb_data, wb_data_this_tr))
            mfg_ts = np.vstack((mfg_ts, fast_apply_mask(img,MFG_mask).mean()))
            hc_ts = np.vstack((hc_ts, fast_apply_mask(img,HC_mask)))
            mc_params = np.vstack((mc_params, np.loadtxt(f'{tmpPath}/temp_aligned.par')))

        '''can only normalize after this'''
        if TR == 30:
            scaling_ref = np.copy(wb_data)
    
    #save the dms, events, and preprocessed image for this run
    for trial_num, _dm in enumerate(dms_out):
        _dm.to_csv(f'{run_outDir}/run-{run}_trial-{trial_num}_dm.csv',index=False)
    
    for trial_num, _events in enumerate(events_out):
        _events.to_csv(f'{run_outDir}/run-{run}_trial-{trial_num}_events.csv',index=False)

    out_img = new_img_like(bidsIncremental.image, out_img, copy_header=True)
    nib.save(out_img, f'{outDir}/{subj}_{sess}_task-feedback_run-{run}_desc-bold_mc_sm.nii.gz')


    print(f"-----------------------------------------------------------------------\n"
    f"END OF RUN {run}! total score = {int(total_score*100)} \n"
    "-----------------------------------------------------------------------")
    bidsInterface.closeStream(streamId)

print("-----------------------------------------------------------------------\n"
"REAL-TIME EXPERIMENT COMPLETE!\n"
"-----------------------------------------------------------------------")
sys.exit(0)
