import os
import sys
import argparse
from subprocess import call
from pathlib import Path
from datetime import datetime, date
from induction_timing import create_induction_timing

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

#find the dicom folder, which needs the datestring in the format of YYYYMMDD
dicomRoot = cfg.dicomRoot
dateString = cfg.dateString
if dateString == "default":
    today = date.today()
    dateString = today.strftime('%Y%m%d')
dicomPath = f'{dicomRoot}/{dateString}.sub_RP{sub:03d}.sub_RP{sub:03d}'
print(f'dicom folder: {dicomPath}')

#this is the output from the processing, where i can also put the files i make here
# outDir = f'/Data1/realtime_outputs/ah7700/rt-press/sub-{subj}'
outRoot = cfg.outRoot
outDir = f'{outRoot}/{subj}/{sess}'
#make the realtime path
os.makedirs(f"{outDir}",exist_ok=True)
print(f'output folder: {outDir}')

print('************************************************')
#create the timing files
print('--- Creating timing files for feedback runs ---')
timing_dict = create_induction_timing(sub,ses)
for run in timing_dict.keys():
    timing_dict[run]['events'].to_csv(f'{outDir}/{subj}_{sess}_task-feedback_run-{run}_events.csv',index=False)
    timing_dict[run]['labels'].to_csv(f'{outDir}/{subj}_{sess}_task-feedback_run-{run}_labels.csv',index=False)

#convert dicoms to nii.gz files
command = f'dcm2niix -z y -f {subj}_%p -o {outDir}/ {dicomPath}/'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"dcm2niix Time: {B-A:.4f}\n")

#there is a non-zero chance that auto-scout images may be included in the real-time files, remove them if neceesary
command = f'rm {outDir}/*Scout*';call(command,shell=True)

#point to some static images needed for registration
standard = f'{projectDir}/standard/MNI152NLin2009cAsym_T1_1mm.nii.gz'
standard_brain = f'{projectDir}/standard/MNI152NLin2009cAsym_T1_1mm_brain.nii.gz'

t1w = f'{outDir}/{subj}_{sess}_anat_t1w.nii.gz'
t1w_brain = f'{outDir}/{subj}_{sess}_t1w_brain.nii.gz'
t1w_brain_mask = f'{outDir}/{subj}_{sess}_t1w_brain_mask.nii.gz'

refseries = f'{outDir}/{subj}_{sess}_task-refseries_bold.nii.gz'
refvol = f'{outDir}/{subj}_{sess}_refvol.nii.gz'
refvol_brain = f'{outDir}/{subj}_{sess}_refvol_brain.nii.gz'
refvol_brain_mask = f'{outDir}/{subj}_{sess}_refvol_brain_mask.nii.gz'

ref2anat = f'{outDir}/refvol_to_anat.mat'
anat2ref = f'{outDir}/anat_to_refvol.mat'
anat2std = f'{outDir}/anat_to_std.mat'
std2anat = f'{outDir}/std_to_anat.mat'
std2ref = f'{outDir}/std_to_refvol.mat'
ref2std = f'{outDir}/refvol_to_std.mat'

#take the mean of the refseries to create the refvol
command = f'fslmaths {refseries} -Tmean {refvol}'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"Refvol creation time: {B-A:.4f}\n")

# skull strip to create a brain mask
command = f'bet {refvol} {refvol_brain} -m -R'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"Skull Strip Time: {B-A:.4f}\n")

#just directly register the refvol to the standard, this time we can use the brain extracted images for cleaner results
command = f'flirt -in {refvol_brain} -ref {standard_brain} -omat {ref2std}'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"FLIRT 1 Time: {B-A:.4f}\n")

#invert it
command = f'convert_xfm -omat {std2ref} -inverse {ref2std}'
A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
print(f"FLIRT 1.5 Time: {B-A:.4f}\n")

#register the masks to subject space
for mask in ['HC','MFG']:
    in_mask = f'{projectDir}/standard/{mask}_1mm.nii.gz'
    out_mask = f'{outDir}/{subj}_{sess}_{mask}_mask.nii.gz'

    command = f'flirt -in {in_mask} -out {out_mask} -ref {refvol_brain} -applyxfm -init {std2ref} -interp nearestneighbour'
    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
    print(f"FLIRT mask 1 Time: {B-A:.4f}\n")

    #mask them with the brain mask just to be careful
    command = f'fslmaths {out_mask} -mas {refvol_brain_mask} {out_mask}'
    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
    print(f"FLIRT mask 2 Time: {B-A:.4f}\n")

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('finished init.py, close the loop!')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')