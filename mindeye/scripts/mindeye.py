"""-----------------------------------------------------------------------------
Imports and set up for mindEye
-----------------------------------------------------------------------------"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import shutil
import json
import argparse
import numpy as np
import math
import time
import random
import string
import h5py
from scipy import stats
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator, DeepSpeedPlugin
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('/home/ri4541@pu.win.princeton.edu/rt-cloud/projects/mindeye/generative_models')
print(os.getcwd())
print(sys.path)
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf
from PIL import Image
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
# custom functions #
import utils_mindeye
sys.path.append('/home/ri4541@pu.win.princeton.edu/rt-cloud/projects/mindeye/models')
from models import *
from cpd_analysis import make_predict_fn, foil_path_for
import pandas as pd
import ants
import nilearn
from nilearn.plotting import plot_design_matrix
import pickle
from collections import defaultdict
import imageio.v2 as imageio
from copy import deepcopy

"""-----------------------------------------------------------------------------
Imports for rtcloud
-----------------------------------------------------------------------------"""
import tempfile
import nibabel as nib
from subprocess import call
from pathlib import Path
from datetime import datetime, date
from scipy.stats import zscore
from nilearn.signal import clean
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import get_data, index_img, concat_imgs, new_img_like
cwd = os.getcwd()
print("cwd ", cwd)
print(os.listdir("projects/mindeye/BidsDir/"))
sys.path.append(cwd)
from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface
from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsRun import BidsRun
from rtCommon.bidsInterface import *

conf_path = 'projects/mindeye/conf/config.json'
try:
    with open(conf_path, 'r') as f:
        config = json.load(f)
    storage_path = config['storage_path']
    data_path = config['data_path']
    derivatives_path = config['derivatives_path']
    output_path = config['output_path']
    fsl_path = config['fsl_path']
    print(f"Using storage path: {storage_path}")
    print(f"Using data path: {data_path}")
    print(f"Using derivatives path: {derivatives_path}")
    print(f"Using output path: {output_path}")
    print(f"Using FSL path: {fsl_path}")
    assert os.path.exists(storage_path), "The specified data and model storage path does not exist."
    assert os.path.exists(data_path), "The specified BOLD path does not exist."
    assert os.path.exists(derivatives_path), "The specified derivatives path does not exist."
    assert os.path.exists(output_path), "The specified output path does not exist."
    assert os.path.exists(fsl_path), "The specified FSL path does not exist."
except FileNotFoundError:
    raise FileNotFoundError("config.json file not found. Please create it with the required paths.")




imsize = 224

### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device

cache_dir= f"{storage_path}/cache"
model_name = "sub-005_ses-01-03_task-C_bs24_MST_rishab_MSTsplit_unionmask_ses-01-03_finetune"
hidden_dim=1024
n_blocks=4 
seq_len = 1
with open(f"{storage_path}/clip_img_embedder", "rb") as input_file:
    clip_img_embedder = pickle.load(input_file)
clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664

num_voxels = 8627
model = utils_mindeye.MindEyeModule(
    num_voxels=num_voxels,
    hidden_dim=hidden_dim,
    seq_len=seq_len,
    clip_emb_dim=clip_emb_dim,
    clip_seq_dim=clip_seq_dim,
    n_blocks=n_blocks,
)
utils_mindeye.count_params(model.ridge)
utils_mindeye.count_params(model.backbone)
utils_mindeye.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

model.build_diffusion_prior(
    clip_emb_dim=clip_emb_dim,
    clip_seq_dim=clip_seq_dim,
    depth=depth,
    dim_head=dim_head,
    heads=heads,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device)

utils_mindeye.count_params(model.diffusion_prior)
utils_mindeye.count_params(model)

# Load pretrained model ckpt
# Replace with pre_trained_fine_tuned_model.pth
tag=f'{model_name}.pth'
outdir = f'{data_path}/model'
# print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
checkpoint = torch.load(outdir+f'/{tag}', map_location='cpu')
state_dict = checkpoint['model_state_dict']
# pdb.set_trace()
model.load_state_dict(state_dict, strict=True)
del checkpoint
# print("ckpt loaded!")

# prep unCLIP
config = OmegaConf.load("projects/mindeye/generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
# first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]
# first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38
with open(f"{storage_path}/diffusion_engine", "rb") as input_file:
    diffusion_engine = pickle.load(input_file)
# set to inference
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)
ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])
batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
    "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
    "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)

sub = "sub-005"
session = "ses-08"
task = 'C'  # 'study' or 'A'; used to search for functional run in bids format
func_task_name = 'C'  # 'study' or 'A'; used to search for functional run in bids format
n_runs = 6

ses_list = [session]
design_ses_list = [session]
    
task_name = f"_task-{task}" if task != 'study' else ''
designdir = f"{data_path}/events"

data, starts, images, is_new_run, image_names, unique_images, len_unique_images = utils_mindeye.load_design_files(
    sub=sub,
    session=session,
    func_task_name=task,
    designdir=designdir,
    design_ses_list=design_ses_list
)

if sub == 'sub-001':
    if session == 'ses-01':
        assert image_names[0] == 'images/image_686_seed_1.png'
    elif session in ('ses-02', 'all'):
        assert image_names[0] == 'all_stimuli/special515/special_40840.jpg'
    elif session == 'ses-03':
        assert image_names[0] == 'all_stimuli/special515/special_69839.jpg'
    elif session == 'ses-04':
        assert image_names[0] == 'all_stimuli/rtmindeye_stimuli/image_686_seed_1.png'
elif sub == 'sub-003':
    assert image_names[0] == 'all_stimuli/rtmindeye_stimuli/image_686_seed_1.png'

unique_images = np.unique(image_names.astype(str))
unique_images = unique_images[(unique_images!="nan")]
len_unique_images = len(unique_images)
print("n_runs",n_runs)

if (sub == 'sub-001' and session == 'ses-04') or (sub == 'sub-003' and session == 'ses-01'):
    assert len(unique_images) == 851

print(image_names[:4])
print(starts[:4])
print(is_new_run[:4])

image_idx = np.array([])  # contains the unique index of each presented image
vox_image_names = np.array([])  # contains the names of the images corresponding to image_idx
all_MST_images = dict()
for i, im in enumerate(image_names):
    # skip if blank, nan
    if im == "blank.jpg":
        i+=1
        continue
    if str(im) == "nan":
        i+=1
        continue
    vox_image_names = np.append(vox_image_names, im)
            
    image_idx_ = np.where(im==unique_images)[0].item()
    image_idx = np.append(image_idx, image_idx_)
    
    all_MST_images[i] = im
    i+=1
    
image_idx = torch.Tensor(image_idx).long()
# for im in new_image_names[MST_images]:
#     assert 'MST_pairs' in im
# assert len(all_MST_images) == 300

unique_MST_images = np.unique(list(all_MST_images.values())) 

MST_ID = np.array([], dtype=int)

vox_idx = np.array([], dtype=int)
for idx, im in enumerate(image_names):  # need unique_MST_images to be defined, so repeating the same loop structure
    # skip if blank, nan
    if im == "blank.jpg" or str(im) == "nan":
        continue
    curr = np.where(im == unique_MST_images)
    if curr[0].size == 0:
        MST_ID = np.append(MST_ID, np.array(len(unique_MST_images)))  # add a value that should be out of range based on the for loop, will index it out later
    else:
        MST_ID = np.append(MST_ID, curr)
        
assert len(MST_ID) == len(image_idx)
print(MST_ID.shape)
if sub == 'sub-005' and session == 'ses-06':
    pass
    # assert len(all_MST_images) == 630
else:
    assert len(all_MST_images) == 693

resize_transform = transforms.Resize((imsize, imsize))
MST_images = []
images = None
for im_name in tqdm(image_idx):
    image_file = f"{unique_images[im_name]}"
    im = imageio.imread(f"{data_path}/{image_file}")
    im = torch.Tensor(im / 255).permute(2,0,1)
    im = resize_transform(im.unsqueeze(0))
    if images is None:
        images = im
    else:
        images = torch.vstack((images, im))
    if ("MST_pairs" in image_file):
        MST_images.append(True)
    else:
        MST_images.append(False)

print("images", images.shape)
MST_images = np.array(MST_images)
print("len MST_images", len(MST_images))
if sub == 'sub-005' and session == 'ses-06':
    pass
elif sub == 'sub-005' and session == 'ses-08':
    assert len(MST_images[MST_images==True]) == 72
else:
    assert len(MST_images[MST_images==True]) == 124
print("MST_images==True", len(MST_images[MST_images==True]))


def get_image_pairs(sub, session, func_task_name, designdir):
    """Loads design files and processes image pairs for a given session."""
    _, _, _, _, image_names, unique_images, _ = utils_mindeye.load_design_files(
        sub=sub,
        session=session,
        func_task_name=func_task_name,
        designdir=designdir,
        design_ses_list=[session]  # Ensure it's a list
    )
    return utils_mindeye.process_images(image_names, unique_images)

all_dicts = []
for s_idx, s in enumerate(ses_list):
    im, vo, _ = get_image_pairs(sub, s, func_task_name, designdir)
    assert len(im) == len(vo)
    all_dicts.append({k:v for k,v in enumerate(vo)})

image_to_indices = defaultdict(lambda: [[] for _ in range(len(ses_list))])
for ses_idx, idx_to_name in enumerate(all_dicts):
    for idx, name in idx_to_name.items():
        image_to_indices[name][ses_idx].append(idx)
        
image_to_indices = dict(image_to_indices)

utils_mindeye.seed_everything(0)
# Collect indices for images containing 'MST_pairs'
MST_idx = [v[0][0] if len(v[0]) > 0 else None for k, v in image_to_indices.items() if 'MST_pairs' in k]

# Remove any None values (in case some images don't have repeats)
MST_idx = [idx for idx in MST_idx if idx is not None]

print("MST_idx", len(MST_idx))

projectDir = os.path.dirname(os.path.realpath(__file__)) #'.../rt-cloud/projects/project_name'
today = date.today()
dateString = today.strftime('%Y%m%d')
today = date.today()
# Month abbreviation, day and year	
d4 = today.strftime("%b-%d-%Y")
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

"""====================REAL-TIME ANALYSIS BELOW====================
===================================================================="""
# clear existing web browser plots if there are any
try:
    webInterface.clearAllPlots()
except:
    pass

# get the mask and the reference files
tmpPath = f"{data_path}/tmp/"  # temporary path to save nifti file at each TR
os.makedirs(tmpPath, exist_ok=True)
ndscore_events = [pd.read_csv(f'{data_path}/events/{sub}_{session}_task-{func_task_name}_run-{run+1:02d}_events.tsv', sep = "\t", header = 0) for run in range(n_runs)]  # create a new list of events_df's which will have the trial_type modified to be unique identifiers
ndscore_tr_labels = [pd.read_csv(f"{data_path}/events/{sub}_{session}_task-{func_task_name}_run-{run+1:02d}_tr_labels.csv") for run in range(n_runs)]
tr_length = 1.5
mask_img = nib.load(f'{data_path}/sub-005_final_mask.nii.gz')  # nsdgeneral mask in functional space
fmriprep_boldref = f"{data_path}/sub-005_ses-01_task-C_run-01_space-T1w_boldref.nii.gz"  # preprocessed boldref from ses-01
rt_vol0 = f"{tmpPath}/vol0.nii.gz" # first volume (vol0000) of real-time session

def fast_apply_mask(target=None,mask=None):
    return target[np.where(mask == 1)].T

fmriprep_boldref_nib = nib.load(fmriprep_boldref)
union_mask = np.load(f"{data_path}/union_mask_from_ses-01-02.npy")

# apply union mask to the nsdgeneral ROI and convert to nifti
assert mask_img.get_fdata().sum() == union_mask.shape
union_mask_img = new_img_like(mask_img, union_mask)

# apply union_mask to mask_img and return nifti object

# Get the data as a boolean array
mask_data = mask_img.get_fdata().astype(bool)

# Flatten only the True voxels in the mask
true_voxel_indices = np.where(mask_data.ravel())[0]

# Apply the union_mask (boolean mask of size 19174)
selected_voxel_indices = true_voxel_indices[union_mask]

# Create a new flattened mask with all False
new_mask_flat = np.zeros(mask_data.size, dtype=bool)

# Set selected voxels to True
new_mask_flat[selected_voxel_indices] = True

# Reshape back to original 3D shape
new_mask_data = new_mask_flat.reshape(mask_data.shape)

# Create new NIfTI image
union_mask_img = nib.Nifti1Image(new_mask_data.astype(np.uint8), affine=mask_img.affine)

def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def convert_image_array_to_PIL(image_array):
    if image_array.ndim == 4:
        image_array = image_array[0]

    # get the dimension to h, w, 3|1
    if image_array.ndim == 3 and image_array.shape[0] == 3:
        image_array = np.transpose(image_array, (1, 2, 0))  # Change shape to (height, width, 3)
    
    # clip the image array to 0-1
    image_array = np.clip(image_array, 0, 1)
    # convert the image array to uint8
    image_array = (image_array * 255).astype('uint8')
    # convert the image array to PIL
    return Image.fromarray(image_array)

plot_images=False
save_individual_images=False
save_all_recons=False
do_cpd=True            # compute CPD scalar (pre-computed embeddings)
do_retrieval=False     # top-5 retrieval images
run_recons=False       # diffusion reconstruction (~4.5s/trial)

# --- real-time setup: build the decoder predict fn, pre-compute CLIP embeddings, and
# --- validate everything the TR loop needs UP FRONT so it can never crash mid-run.
predict_fn = make_predict_fn(model, device)  # betas -> predicted CLIP clip_voxels (ridge+backbone)

# every MST label the loop will read across all runs (authoritative source = tr_labels)
all_mst_labels = sorted({lab for trl in ndscore_tr_labels
                         for lab in trl["tr_label_shifted"].astype(str) if "MST_pairs" in lab})
foil_name_for = {lab: os.path.relpath(foil_path_for(os.path.join(data_path, lab)), data_path)
                 for lab in all_mst_labels}  # raises now if any foil is missing/ambiguous

clip_embeds = {}
if do_cpd:
    needed = sorted(set(all_mst_labels) | set(foil_name_for.values()))
    for name in needed:
        assert os.path.exists(os.path.join(data_path, name)), f"missing stimulus on disk: {name}"
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for name in needed:
            im = imageio.imread(os.path.join(data_path, name))
            im = resize_transform(torch.Tensor(im / 255).permute(2, 0, 1).unsqueeze(0)).to(device)
            clip_embeds[name] = clip_img_embedder(im).float().cpu()
    print(f"pre-computed {len(clip_embeds)} CLIP embeddings (MST stimuli + foils)")

# ---- pre-flight assertions: fail before the real-time loop, never during it ----
assert do_cpd or do_retrieval or run_recons, "enable at least one of do_cpd/do_retrieval/run_recons"
# trial-N naming assumes every stimulus image is an MST pairmate (dense 1..N index); fail
# loudly here if a future session mixes in non-MST stimuli so the indexing isn't silently wrong
non_mst_labels = sorted({lab for trl in ndscore_tr_labels
                         for lab in trl["tr_label_shifted"].astype(str)
                         if lab not in ('blank', 'blank.jpg') and "MST_pairs" not in lab})
assert not non_mst_labels, \
    f"non-MST stimulus labels present; trial-N indexing assumes MST-only stimuli: {non_mst_labels}"
# every MST trial resolves its ground-truth image via vox_image_names, regardless of flags
for lab in all_mst_labels:
    assert lab in vox_image_names, f"MST label not in vox_image_names (no ground-truth row): {lab}"
if do_cpd:
    for lab in all_mst_labels:
        assert lab in clip_embeds and foil_name_for[lab] in clip_embeds, \
            f"do_cpd: no pre-computed embedding for {lab} or its foil"
if do_retrieval:
    assert len(MST_idx) > 0, "do_retrieval: empty MST retrieval pool (MST_idx)"
if run_recons:
    assert model.diffusion_prior is not None, "run_recons: diffusion_prior not initialized"

mc_dir = f"{derivatives_path}/motion_corrected"
mc_resampled_dir = f"{derivatives_path}/motion_corrected_resampled"
if os.path.exists(mc_dir):
    shutil.rmtree(mc_dir)
os.makedirs(mc_dir)
if os.path.exists(mc_resampled_dir):
    shutil.rmtree(mc_resampled_dir)
os.makedirs(mc_resampled_dir)

rt_to_fmriprep_mat = f'{derivatives_path}/rtref_to_ses1ref'
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
assert np.all(fmriprep_boldref_nib.affine == union_mask_img.affine)
all_betas = []
shown_filenames = dict()

# session-level accumulators (in-memory) for the always-on end-of-session evaluation
session_cpd, session_clipvoxels, session_ground_truth = [], [], []
session_recons, session_retrieved = [], []

# go through each run
for run_num in range(1, n_runs + 1):
    print(f"Start of real-time session run {run_num}!\n")
    cwd = os.getcwd()
    print("cwd ", cwd)
    # print(os.listdir(f"{data_path}/raw_bids"))
    run_to_dicom = {1:5, 2:6, 3:7, 4:8, 5:10, 6:11, 7:12, 8:13, 9:15, 10:16, 11:17}
    
    # dicomNamePattern = "{RUN}-{TR}-1.dcm"  # use this for scanner dicoms WITHOUT "RT Start On"
    dicomNamePattern = "001_{RUN:06d}_{TR:06d}.dcm"  # use this for scanner dicoms WITH "RT Start On"
    
    dicomScanNamePattern = stringPartialFormat(dicomNamePattern, 'RUN', run_to_dicom[run_num])

    dicom_filename = "phantom2"  # when registering the subject into the scanner, this is what was entered for last name and subject ID
    dicomDir = f"/home/scontrol/20260618.{dicom_filename}.{dicom_filename}"  # directory to use when the scanner mounts to the real-time computer
    # dicomDir = f"{data_path}/dicom_ses-03"
    streamID = bidsInterface.initDicomBidsStream(dicomDir, dicomScanNamePattern,
                                               300000, anonymize=False,
                                               **{'subject':f'{dicom_filename}',
                                                  'run':f'{run_num}',
                                                  'task':'C'})

    print(f"Run {run_num} started")
    mc_params = []
    imgs = []
    events_df = ndscore_events[run_num - 1]
    tr_labels_shifted = ndscore_tr_labels[run_num - 1]["tr_label_shifted"].tolist()
    events_df = events_df[events_df['image_name'] != 'blank.jpg']  # must drop blank.jpg after tr_labels_shifted is defined to keep indexing consistent
    beta_maps_list = []
    all_trial_names_list = []
    all_images = None

    save_path = f"{output_path}/{sub}_{session}_task-{func_task_name}_run-{run_num:02d}_recons"
    os.makedirs(save_path, exist_ok=True)
    if save_individual_images:
        os.makedirs(os.path.join(save_path, "individual_images"), exist_ok=True)

    all_recons_save = []
    all_clipvoxels_save = []
    all_ground_truth_save = []
    all_retrieved_save = []
    all_cpd_save = []

    stimulus_trial_counter = 0
    # Counter for MST_pairs trials and evenly spaced recon points
    # mst_trial_counter = 0
    # mst_total = 63  # total MST_pairs trials in a run (adjust if needed)
    # mst_recon_points = np.linspace(5, mst_total, 7, dtype=int).tolist()
    T1_brain = f"{data_path}/{sub}_desc-preproc_T1w_brain.nii.gz"
    n_trs = 288
    assert len(tr_labels_shifted) == n_trs, "there should be image labels for each TR"
    assert all(label in image_names for label in tr_labels_shifted if label != 'blank'), "Some labels in tr_labels_shifted are missing from image_names."
    assert len(images) > n_trs, "images array is too short."

    for TR in range(n_trs):
        print(f"TR {TR}")
        incremental_bids_image = bidsInterface.getIncremental(streamID,volIdx=TR+1,
                                        timeout=999999,demoStep=0)
        image_data = incremental_bids_image.image
        curr_nifti = f'{tmpPath}/temp.nii'
        nib.save(image_data, curr_nifti)

        current_label = tr_labels_shifted[TR]
        print(current_label)
        
        if TR == 0 and run_num == 1:
            nib.save(image_data, rt_vol0)  # real-time volume 0, will be used to motion correct all future volumes

            os.system(f"flirt -in {rt_vol0} \
                -ref {fmriprep_boldref} \
                -omat {rt_to_fmriprep_mat} \
                -dof 6")  # register real-time volume 0 to the fmriprep bold reference image and output the corresponding transformation matrix

        mc = f"{tmpPath}/temp_aligned"
        os.system(f"{fsl_path}/mcflirt -in {curr_nifti} -reffile {rt_vol0} -out {mc} -plots -mats")
        mc_params.append(np.loadtxt(f'{mc}.par'))

        current_tr_to_orig_ses = f"{derivatives_path}/current_tr_to_orig_ses_run{run_num}"
        os.system(f"convert_xfm -concat {rt_to_fmriprep_mat} -omat {current_tr_to_orig_ses} {mc}.mat/MAT_0000")  # combine 2 transforms: motion correction and cross-session registration
        
        final_vol = f"{mc_resampled_dir}/{sub}_{session}_run-{run_num:02d}_{TR:04d}_mc_boldres.nii.gz"
        os.system(f"flirt -in {curr_nifti} \
            -ref {fmriprep_boldref} \
            -out {final_vol} \
            -init {current_tr_to_orig_ses} \
            -applyxfm")  # apply combined transformation matrix to the current TR

        os.system(f"rm -r {mc}.mat")
        imgs.append(get_data(final_vol))

        if current_label not in ('blank', 'blank.jpg'):
            events_df = events_df.copy()
            events_df['onset'] = events_df['onset'].astype(float)

            run_start_time = events_df['onset'].iloc[0]
            events_df = events_df.copy()
            events_df['onset'] -= run_start_time

            cropped_events = events_df[events_df.onset <= TR*tr_length]
            cropped_events = cropped_events.copy()
            cropped_events.loc[:, 'trial_type'] = np.where(cropped_events['trial_number'] == stimulus_trial_counter, "probe", "reference")
            cropped_events = cropped_events.drop(columns=['is_correct', 'image_name', 'response_time', 'trial_number'])

            # collect all of the images at each TR into a 4D time series
            img = np.rollaxis(np.array(imgs),0,4)
            img = new_img_like(fmriprep_boldref_nib,img,copy_header=True)
            # run the model with mc_params confounds to motion correct
            lss_glm = FirstLevelModel(t_r=tr_length,slice_time_ref=0,hrf_model='glover',
                        drift_model='cosine', drift_order=1,high_pass=0.01,mask_img=union_mask_img,
                        signal_scaling=False,smoothing_fwhm=None,noise_model='ar1',
                        n_jobs=-1,verbose=-1,memory_level=1,minimize_memory=True)
            
            lss_glm.fit(run_imgs=img, events=cropped_events, confounds = pd.DataFrame(np.array(mc_params)))
            dm = lss_glm.design_matrices_[0]
            # get the beta map and mask it
            beta_map = lss_glm.compute_contrast("probe", output_type="effect_size")
            beta_map_np = beta_map.get_fdata()
            beta_map_np = fast_apply_mask(target=beta_map_np,mask=union_mask_img.get_fdata())
            all_betas.append(beta_map_np)
            print('all_betas shape:', np.array(all_betas).shape)
            
            if current_label not in shown_filenames.keys():
                shown_filenames[current_label] = [len(all_betas)]
                is_repeat = False
            else:
                shown_filenames[current_label].append(len(all_betas))
                is_repeat = True
                print(f"The following image is a repeat!\n{shown_filenames[current_label]}")

            if "MST_pairs" in current_label:  # and run_num >= 2:
                # mst_trial_counter += 1
                # if mst_trial_counter in mst_recon_points:
                correct_image_index = np.where(current_label == vox_image_names)[0][0]  # using the first occurrence based on image name, assumes that repeated images are identical (which they should be)
                z_mean = np.mean(np.array(all_betas), axis=0)
                z_std = np.std(np.array(all_betas), axis=0)
                # if is_repeat:
                #     beta_repeat_idxs = shown_filenames[current_label]
                #     assert len(beta_repeat_idxs) > 1  # this image has been shown more than once
                #     betas_repeats = []
                #     for b in beta_repeat_idxs:
                #         print(f"Averaging over {len(beta_repeat_idxs)} repeats")
                #         # re-z-score the older betas in addition to the newest beta since we have more data to z-score with
                #         tmp = ((np.array(all_betas) - z_mean) / (z_std + 1e-6))[b-1]
                #         betas_repeats.append(tmp)
                #     betas = np.mean(np.array(betas_repeats), axis=0)  # average beta patterns over all available repeats
                # else:
                betas = ((np.array(all_betas) - z_mean) / (z_std + 1e-6))[-1]  # use only the beta pattern from the most recent image
                betas = betas[np.newaxis, np.newaxis, :]
                betas_tt = torch.Tensor(betas).to("cpu")
                # predicted CLIP embedding (ridge+backbone only), decoupled from diffusion recon
                clipvoxelsTR = predict_fn(betas_tt)
                if run_recons:
                    reconsTR, _ = utils_mindeye.do_reconstructions(
                        model,
                        betas_tt,
                        diffusion_engine,
                        vector_suffix,
                        imsize,
                        device,
                    )

                # assemble the result dict from whichever options are enabled
                values_dict = {}
                cpd_val = None
                if do_cpd:
                    foil_label = foil_name_for[current_label]
                    cpd_val = utils_mindeye.cpd_from_embeddings(
                        pred=clipvoxelsTR,
                        correct=clip_embeds[current_label],
                        foil=clip_embeds[foil_label],
                    )
                    values_dict["cpd"] = cpd_val
                if do_retrieval:
                    values_dict.update(utils_mindeye.do_retrievals(
                        clip_img_embedder,
                        clipvoxelsTR,
                        all_images=images[MST_idx],
                        imsize=imsize,
                        device=device,
                        total_retrievals=5,
                    ))
                if run_recons:
                    values_dict["recons"] = utils_mindeye.compress_and_encode_image((reconsTR.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy())
                    reconsTR = reconsTR.half().numpy()

                resized = transforms.Resize((imsize, imsize), antialias=True)(images[correct_image_index])
                encoded_image = utils_mindeye.compress_and_encode_image(
                    (resized.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy())
                values_dict["ground_truth"] = encoded_image

                # subjInterface.setResultDict allows us to send to the analysis listener immediately
                # name by session-wide image index, 1-indexed (all_betas was just appended for this trial)
                subjInterface.setResultDict(name=f'trial-{len(all_betas)}',
                                            values=values_dict)

                # recon image (only when reconstruction was run)
                if run_recons:
                    image_array = reconsTR[0]
                    # If the image has 3 channels (RGB), reorder to (H, W, 3)
                    if image_array.ndim == 3 and image_array.shape[0] == 3:
                        image_array = np.transpose(image_array, (1, 2, 0))

                # Display the image
                if plot_images:
                    n_panels = 1 + (1 if run_recons else 0) + (5 if do_retrieval else 0)
                    col = 1
                    plt.figure(figsize=(2.5 * n_panels, 5))
                    plt.subplot(1, n_panels, col); col += 1
                    plt.title("Original Image")
                    plt.imshow(images[correct_image_index].half().numpy().transpose(1, 2, 0), cmap='gray')
                    plt.axis('off')
                    if run_recons:
                        plt.subplot(1, n_panels, col); col += 1
                        plt.title("Reconstructed Image")
                        plt.imshow(image_array, cmap='gray' if image_array.ndim == 2 else None)
                        plt.axis('off')
                    if do_retrieval:
                        for i in range(5):
                            plt.subplot(1, n_panels, col); col += 1
                            plt.title(f"Retrieval {i+1}")
                            plt.imshow(np.array(values_dict[f"attempt{i+1}"][0]).transpose(1, 2, 0), cmap='gray')
                            plt.axis('off')
                    plt.show()

                # save reconstructed/retrieved images, clip_voxels, and ground truth image
                if save_individual_images:
                    if run_recons:
                        convert_image_array_to_PIL(image_array).save(os.path.join(save_path, "individual_images", f"run{run_num}_TR{TR}_reconstructed.png"))
                    if do_retrieval:
                        for key, value in values_dict.items():
                            if key.startswith("attempt"):
                                convert_image_array_to_PIL(np.array(value)).save(os.path.join(save_path, "individual_images", f"run{run_num}_TR{TR}_retrieved_{key}.png"))
                    # save the clip_voxels
                    np.save(os.path.join(save_path, "individual_images", f"run{run_num}_TR{TR}_clip_voxels.npy"), clipvoxelsTR)
                    # save the ground truth image
                    convert_image_array_to_PIL(images[correct_image_index].half().numpy()).save(os.path.join(save_path, "individual_images", f"run{run_num}_TR{TR}_ground_truth.png"))

                # accumulate per-run results (only what each enabled option produced)
                if do_cpd:
                    all_cpd_save.append(cpd_val)
                if run_recons:
                    all_recons_save.append(image_array)
                if do_retrieval:
                    all_retrieved_save.append([np.array(value) for key, value in values_dict.items() if key.startswith("attempt")])
                all_clipvoxels_save.append(clipvoxelsTR)
                all_ground_truth_save.append(images[correct_image_index].half().numpy())
                # else:
                #     subjInterface.setResultDict(name=f'run{run_num}_TR{TR}',
                #         values={'pass': "pass"})

            else:
                pass
                # subjInterface.setResultDict(name=f'run{run_num}_TR{TR}',
                #     values={'pass': "pass"})
            
            stimulus_trial_counter += 1
        elif current_label == 'blank.jpg':
            pass
            # subjInterface.setResultDict(name=f'run{run_num}_TR{TR}',
            #     values={'pass': "pass"})
            stimulus_trial_counter += 1
        else:
            assert current_label == 'blank'
            # blank TR
            # when we are not at the end of a stimulus trial, send an empty dictionary to the analysis listener with "pass"
            # subjInterface.setResultDict(name=f'run{run_num}_TR{TR}',
            #                 values={'pass': "pass"})
        
    print(f"==END OF RUN {run_num}!==\n")

    # save the design matrix for the current run
    dm.to_csv(os.path.join(save_path, f"design_run-{run_num:02d}.csv"))
    plot_design_matrix(dm, output_file=os.path.join(save_path, "dm"))
    dm[['probe', 'reference']].plot(title='Probe/Reference Regressors', figsize=(10, 4))
    plt.savefig(os.path.join(save_path, "regressors"))
    # save betas so far
    np.save(os.path.join(save_path, f"betas_run-{run_num:02d}.npy"), np.array(all_betas))
    print(f"==END OF RUN {run_num}!==\n")
    # always save the base data (clipvoxels + ground truth; betas are already saved above)
    # so reconstructions / top-1 retrieval / CPD can be recomputed later regardless of flags
    if len(all_clipvoxels_save) > 0:
        all_clipvoxels_save_tensor = torch.stack(all_clipvoxels_save, dim=0)
        all_ground_truth_save_tensor = torch.tensor(all_ground_truth_save)
        torch.save(all_clipvoxels_save_tensor, os.path.join(save_path, "all_clipvoxels.pt"))
        torch.save(all_ground_truth_save_tensor, os.path.join(save_path, "all_ground_truth.pt"))
        print("all_clipvoxels_save_tensor.shape: ", all_clipvoxels_save_tensor.shape)
        print("all_ground_truth_save_tensor.shape: ", all_ground_truth_save_tensor.shape)
        if do_cpd:
            torch.save(torch.tensor(all_cpd_save), os.path.join(save_path, "all_cpd.pt"))

        # heavy derived outputs: only persisted when produced AND save_all_recons is on
        if save_all_recons and run_recons:
            all_recons_save_tensor = torch.tensor(all_recons_save).permute(0,3,1,2)
            torch.save(all_recons_save_tensor, os.path.join(save_path, "all_recons.pt"))
            print("all_recons_save_tensor.shape: ", all_recons_save_tensor.shape)
        if save_all_recons and do_retrieval:
            all_retrieved_save_tensor = torch.stack([torch.tensor(np.array(item)) for item in all_retrieved_save], dim=0)
            torch.save(all_retrieved_save_tensor, os.path.join(save_path, "all_retrieved.pt"))
            print("all_retrieved_save_tensor.shape: ", all_retrieved_save_tensor.shape)
        print("Tensors saved successfully on ", save_path)

    # accumulate this run's results into the session-level lists for end-of-session eval
    session_cpd.extend(all_cpd_save)
    session_clipvoxels.extend(all_clipvoxels_save)
    session_ground_truth.extend(all_ground_truth_save)
    session_recons.extend(all_recons_save)
    session_retrieved.extend(all_retrieved_save)

    bidsInterface.closeStream(streamID)


print('all done!')
# ---- end-of-session evaluation (always runs; adapts to the enabled options) ----
from utils_mindeye import (calculate_retrieval_metrics, calculate_alexnet, calculate_clip,
                           calculate_swav, calculate_efficientnet_b1, calculate_inception_v3,
                           calculate_pixcorr, calculate_ssim, deduplicate_tensors)

metrics = {}

# CPD + pairmate 2-AFC (a 2-AFC trial is correct <=> CPD > 0 for L2-normalized embeddings,
# equivalent to cpd_analysis.pairmate_2afc_accuracy)
if do_cpd and len(session_cpd) > 0:
    cpd_arr = np.array(session_cpd, dtype=np.float32)
    metrics["meanCPD"] = float(cpd_arr.mean())
    metrics["pairmate_2afc"] = float((cpd_arr > 0).mean())
    print(f"meanCPD={metrics['meanCPD']:+.4f}  pairmate_2afc={metrics['pairmate_2afc']:.3f}  (n={len(cpd_arr)})")

# forward/backward retrieval from predicted CLIP embeddings (needs repeated images)
if do_retrieval and len(session_clipvoxels) > 0:
    try:
        clipvoxels_t = torch.stack([torch.as_tensor(np.array(c)) for c in session_clipvoxels]).to(torch.float16).to(device)
        ground_truth_t = torch.stack([torch.as_tensor(np.array(g)) for g in session_ground_truth]).to(torch.float16).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, _, duplicated = deduplicate_tensors(clipvoxels_t, ground_truth_t)
            dup = np.array(duplicated)
            metrics["fwd_retrieval_subset0"], metrics["bwd_retrieval_subset0"] = \
                calculate_retrieval_metrics(clipvoxels_t[dup[:, 0]], ground_truth_t[dup[:, 0]])
            metrics["fwd_retrieval_subset1"], metrics["bwd_retrieval_subset1"] = \
                calculate_retrieval_metrics(clipvoxels_t[dup[:, 1]], ground_truth_t[dup[:, 1]])
    except Exception as e:
        print(f"retrieval eval skipped: {e}")

# reconstruction-quality metrics (only when reconstructions were generated)
if run_recons and len(session_recons) > 0:
    try:
        recons_t = torch.tensor(np.array(session_recons)).permute(0, 3, 1, 2).to(torch.float16).to(device)
        ground_truth_t = torch.stack([torch.as_tensor(np.array(g)) for g in session_ground_truth]).to(torch.float16).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            metrics["pixcorr"] = calculate_pixcorr(recons_t, ground_truth_t)
            metrics["ssim"] = calculate_ssim(recons_t, ground_truth_t)
            metrics["alexnet2"], metrics["alexnet5"] = calculate_alexnet(recons_t, ground_truth_t)
            metrics["inception"] = calculate_inception_v3(recons_t, ground_truth_t)
            metrics["clip_"] = calculate_clip(recons_t, ground_truth_t)
            metrics["efficientnet"] = calculate_efficientnet_b1(recons_t, ground_truth_t)
            metrics["swav"] = calculate_swav(recons_t, ground_truth_t)
    except Exception as e:
        print(f"reconstruction eval skipped: {e}")

# print + save whatever was computed
if metrics:
    df_metrics = pd.DataFrame({"Metric": list(metrics.keys()),
                               "Value": [float(v) for v in metrics.values()]})
    print(df_metrics.to_string(index=False))
    df_metrics.to_csv(os.path.join(output_path, f"{sub}_{session}_metrics.csv"), index=False)
else:
    print("no evaluation metrics computed for the enabled options")