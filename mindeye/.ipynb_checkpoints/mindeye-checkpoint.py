#!/usr/bin/env python
# coding: utf-8

# In[1]:


# set up main path where everything will be you should download the
# hugging face directory described in readme and put it here on the same
# server where the data analyzer is run so that the data analyzer code with 
# the GPU can access these files
# You should replace the below path with your location
data_and_model_storage_path = '/scratch/gpfs/ri4541/rt_mindEye/rt_all_data/'
"""-----------------------------------------------------------------------------
Imports and set up for mindEye
-----------------------------------------------------------------------------"""
err
import os
import sys
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
sys.path.append('generative_models/')
# print(sys.path)
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
# custom functions #
import utils_mindeye
from models import *
### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device


# In[2]:


cache_dir= f"{data_and_model_storage_path}cache"
model_name="multisubject_subj01_1024hid_nolow_300ep_milestone2"
subj=1
hidden_dim=1024
blurry_recon = False
n_blocks=4 
seq_len = 1

import pickle
with open(f"{data_and_model_storage_path}clip_img_embedder", "rb") as input_file:
    clip_img_embedder = pickle.load(input_file)
clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664


# In[3]:


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

model = MindEyeModule()

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(seq_len)], dim=1)
        return out
num_voxels = 25225
model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim, seq_len=seq_len)

from diffusers.models.vae import Decoder
class BrainNetwork(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=n_blocks, drop=.15, 
                clip_size=768):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size

        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])

        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)


    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )

    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )

    def forward(self, x):
        # make empty tensors
        c,b,t = torch.Tensor([0.]), torch.Tensor([[0.],[0.]]), torch.Tensor([0.])

        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)

            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)

        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        c = self.clip_proj(backbone)

        return backbone, c, b

model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=seq_len, 
                        clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim) 
utils_mindeye.count_params(model.ridge)
utils_mindeye.count_params(model.backbone)
utils_mindeye.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device)

utils_mindeye.count_params(model.diffusion_prior)
utils_mindeye.count_params(model)


# In[4]:


# Load pretrained model ckpt
# Replace with pre_trained_fine_tuned_model.pth
tag='pretrained_fine-tuned_sliceTimed0.5.pth'
outdir = os.path.abspath(f'{data_and_model_storage_path}')

# print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir+f'/{tag}', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except: # probably ckpt is saved using deepspeed format
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
# print("ckpt loaded!")


# In[5]:


# prep unCLIP
config = OmegaConf.load("generative_models/configs/unclip6.yaml")
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
with open(f"{data_and_model_storage_path}diffusion_engine", "rb") as input_file:
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
f = h5py.File(f'{data_and_model_storage_path}coco_images_224_float16.hdf5', 'r')
images = f['images']


# In[6]:


import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import *
from nilearn.image import get_data, index_img, concat_imgs, new_img_like


# get the mask and the reference files
ndscore_events = [pd.read_csv(f'{data_and_model_storage_path}sub-01_ses-nsd02_task-nsdcore_run-{run:02d}_events.tsv', sep = "\t", header = 0) for run in range(1,2)]# create a new list of events_df's which will have the trial_type modified to be unique identifiers
ndscore_tr_labels = [pd.read_csv(f"{data_and_model_storage_path}sub-01_ses-nsd02_task-nsdcore_run-{run_num:02d}_tr_labels.csv") for run_num in range(1,2)]
tr_length = 1.6
mask_img = nib.load(f'{data_and_model_storage_path}sub-01_nsdgeneral_to_day1ref.nii.gz')
day1_boldref= f"{data_and_model_storage_path}day1_bold_ref.nii.gz" #day 1 reference image is the middle volume (vol0094) of day1run1
day2_boldref= f"{data_and_model_storage_path}day2_bold_ref.nii.gz" #day 2 reference image is the first volume (vol0000) of day2
day2_to_day1_mat =  f"{data_and_model_storage_path}day2ref_to_day1ref"
def fast_apply_mask(target=None,mask=None):
    return target[np.where(mask == 1)].T
lss_glm = FirstLevelModel(t_r=tr_length,slice_time_ref=0.5,hrf_model='glover',
                        drift_model=None,high_pass=None,mask_img=mask_img,
                        signal_scaling=False,smoothing_fwhm=None,noise_model='ar1',
                        n_jobs=-1,verbose=-1,memory_level=1,minimize_memory=True)
day1_boldref_nibd = nib.load(day1_boldref)


# In[7]:


def do_reconstructions(betas_tt):
    """
    takes in the beta map for a stimulus trial in torch tensor format (tt)

    returns reconstructions and clipvoxels for retrievals
    """
    # start_reconstruction_time = time.time()
    model.to(device)
    model.eval().requires_grad_(False)
    clipvoxelsTR = None
    reconsTR = None
    num_samples_per_image = 1
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        voxel = betas_tt
        voxel = voxel.to(device)
        voxel_ridge = model.ridge(voxel[:,[0]],0) # 0th index of subj_list
        backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
        clip_voxels = clip_voxels0
        backbone = backbone0
        blurry_image_enc = blurry_image_enc0[0]
        clipvoxelsTR = clip_voxels.cpu()
        prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                        text_cond = dict(text_embed = backbone), 
                        cond_scale = 1., timesteps = 20)  
        for i in range(len(voxel)):
            samples = utils_mindeye.unclip_recon(prior_out[[i]],
                            diffusion_engine,
                            vector_suffix,
                            num_samples=num_samples_per_image)
            if reconsTR is None:
                reconsTR = samples.cpu()
            else:
                reconsTR = torch.vstack((reconsTR, samples.cpu()))
            imsize = 224
            reconsTR = transforms.Resize((imsize,imsize), antialias=True)(reconsTR).float().numpy().tolist()
        return reconsTR, clipvoxelsTR
    
def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def get_top_retrievals(clipvoxel, all_images, stimulus_trial_counter):
    '''
    clipvoxel: output from do_recons that contains that information needed for retrievals
    all_images: all ground truth actually seen images by the participant in day 2 run 1

    outputs the top retrievals
    '''
    values_dict = {}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        emb = clip_img_embedder(torch.reshape(all_images,(all_images.shape[0], 3, 224, 224)).to(device)).float() # CLIP-Image
        emb = emb.cpu()
        emb_ = clipvoxel # CLIP-Brain
        emb = emb.reshape(len(emb),-1)
        emb_ = np.reshape(emb_, (1, 425984))
        emb = nn.functional.normalize(emb,dim=-1)
        emb_ = nn.functional.normalize(emb_,dim=-1)
        emb_ = emb_.float()
        fwd_sim = batchwise_cosine_similarity(emb_,emb)  # brain, clip
        print("Given Brain embedding, find correct Image embedding")
    fwd_sim = np.array(fwd_sim.cpu())
    imsize = 224
    values_dict["ground_truth"] = transforms.Resize((imsize,imsize), antialias=True)(all_images[stimulus_trial_counter]).float().numpy().tolist()
   # values_dict["ground_truth"] = all_images[stimulus_trial_counter].numpy().tolist()
    for attempt in range(5):
        which = np.flip(np.argsort(fwd_sim, axis = 0))[attempt]
       # values_dict[f"attempt{(attempt+1)}"] = all_images[which.copy()].numpy().tolist()
        values_dict[f"attempt{(attempt+1)}"] = transforms.Resize((imsize,imsize), antialias=True)(all_images[which.copy()]).float().numpy().tolist()
    return values_dict


# In[8]:


# get the mask and the reference files
ndscore_events = [pd.read_csv(f'{data_and_model_storage_path}sub-01_ses-nsd02_task-nsdcore_run-{run:02d}_events.tsv', sep = "\t", header = 0) for run in range(1,2)]# create a new list of events_df's which will have the trial_type modified to be unique identifiers
ndscore_tr_labels = [pd.read_csv(f"{data_and_model_storage_path}sub-01_ses-nsd02_task-nsdcore_run-{run_num:02d}_tr_labels.csv") for run_num in range(1,2)]
tr_length = 1.6
mask_img = nib.load(f'{data_and_model_storage_path}sub-01_nsdgeneral_to_day1ref.nii.gz')
day1_boldref= f"{data_and_model_storage_path}day1_bold_ref.nii.gz" #day 1 reference image is the middle volume (vol0094) of day1run1
day2_boldref= f"{data_and_model_storage_path}day2_bold_ref.nii.gz" #day 2 reference image is the first volume (vol0000) of day2
day2_to_day1_mat =  f"{data_and_model_storage_path}day2ref_to_day1ref"
def fast_apply_mask(target=None,mask=None):
    return target[np.where(mask == 1)].T
lss_glm = FirstLevelModel(t_r=tr_length,slice_time_ref=0.5,hrf_model='glover',
                        drift_model=None,high_pass=None,mask_img=mask_img,
                        signal_scaling=False,smoothing_fwhm=None,noise_model='ar1',
                        n_jobs=-1,verbose=-1,memory_level=1,minimize_memory=True)
day1_boldref_nibd = nib.load(day1_boldref)


# In[ ]:


run_num = 1
print(f"{run_num} started")
mc_params = []
imgs = []
events_df = ndscore_events[run_num - 1]
tr_labels_hrf = ndscore_tr_labels[run_num - 1]["tr_label_hrf"].tolist()
beta_maps_list = []
all_trial_names_list = []
# get the all images tensor
all_images = None
seen_label_before = ["blank"]
# get the list of all images in torch tensor format for this run (should be 62 or 63 images)
all_COCO_ids = []
for TR in range(186):
    if tr_labels_hrf[TR] not in seen_label_before:
        seen_label_before.append(tr_labels_hrf[TR])
        image_COCO_id = int(float(tr_labels_hrf[TR].split("_")[1])) - 1
        new_image_pt = torch.from_numpy(np.reshape(images[image_COCO_id],(1,3,224,224)))
        all_images = new_image_pt if all_images == None else torch.vstack((all_images, new_image_pt))
        all_COCO_ids.append(image_COCO_id)
# print(all_COCO_ids)

stimulus_trial_counter = 0
bold = nib.load("/home/ri4541/rt-cloud/projects/mindeye/BidsDir/sub-01/ses-nsd02/func/sub-01_ses-nsd02_task-nsdcore_run-01_bold.nii.gz")
for TR in range(188):
    print(f"TR {TR}")
    # stream in the nifti
    image_data = bold.slicer[:,:,:,None,TR]  # None prevents the final dimension from being dropped because it's a singleton; nibabel methods expect a 4D array
    current_label = tr_labels_hrf[TR]
    if TR == 0:
        day2_run1_bold_ref = image_data
        # make the day 2 bold ref
        nib.save(image_data, day2_boldref)
        # save the transformation from the day 2 bold ref to the day 1 
        os.system(f"flirt -in {day2_boldref} \
        -ref {day1_boldref} \
        -omat {day2_to_day1_mat} \
        -dof 6")
    # load nifti file
    tmp = f'{data_and_model_storage_path}day2_subj1/tmp_run{run_num}.nii.gz'
    nib.save(index_img(image_data,0),tmp)
    start = time.time()
    # on first tr the motion correction will have no issue so that mc_params is properly populated
    mc = f'{data_and_model_storage_path}day2_subj1/tmp_mc_run{run_num}'
    os.system(f"mcflirt -in {tmp} -reffile {day2_boldref} -out {mc} -plots -mats")
    mc_params.append(np.loadtxt(f'{mc}.par'))

    slice_timed = f'{data_and_model_storage_path}day2_subj1/tmp_sT_run{run_num}'
    slice_tcustom_path = f'{data_and_model_storage_path}slice_timing_day2_run1.txt'
    os.system(f"slicetimer -i {tmp} -o {slice_timed} --tcustom={slice_tcustom_path}")

    mc_day1_aligned = f'{data_and_model_storage_path}day2_subj1/tmp_mc_day1_aligned_run{run_num}'
    current_tr_to_day1 = f"{data_and_model_storage_path}day2_subj1/current_tr_to_day1_run{run_num}"
    os.system(f"convert_xfm -concat {day2_to_day1_mat} -omat {current_tr_to_day1} {mc}.mat/MAT_0000")    
    # apply concatenated matrix to the current TR
    os.system(f"flirt -in {slice_timed} \
    -ref {day1_boldref} \
    -out {mc_day1_aligned} \
    -init {current_tr_to_day1} \
    -applyxfm")
    # now delete the mc from current tr to bold reference mat
    os.system(f"rm -r {mc}.mat") 
    imgs.append(get_data(mc_day1_aligned + ".nii.gz")) # only add to imgs list
    if tr_labels_hrf[TR] != tr_labels_hrf[TR + 1] and tr_labels_hrf[TR] != "blank":
        cropped_events = events_df[events_df.trial_number <= int(float(tr_labels_hrf[TR].split("_")[3]))].astype(str)
        for i_trial, trial in cropped_events.iterrows():
            cropped_events.loc[i_trial, "trial_type"] = "reference" if i_trial < (len(cropped_events) - 1) else "probe"
            
        cropped_events = cropped_events.drop(columns=['total_novel_presses', 'change_mind', 'is_correct', 'time', 
                                              'response_time', 'response', '73k_id', 'trial_number', 
                                              '10k_id', 'memory_first', 'is_old_session', 'is_correct_session', 
                                              'missing_data', 'total_old_presses', 'memory_recent'])

        # get the image id from this stimulus trial that we are fitting a model on
        image_COCO_id = int(float(tr_labels_hrf[TR].split("_")[1])) - 1
        # collect all of the images at each TR into a 4D time series
        img = np.rollaxis(np.array(imgs),0,4)
        img = new_img_like(day1_boldref_nibd,img,copy_header=True)
        # run the model with mc_params confounds to motion correct
        lss_glm.fit(run_imgs=img,events=cropped_events, confounds = pd.DataFrame(np.array(mc_params)))
        # get the beta map and mask it
        beta_map = lss_glm.compute_contrast("probe", output_type="effect_size")
        beta_map_np = beta_map.get_fdata()
        beta_map_np = fast_apply_mask(target=beta_map_np,mask=mask_img.get_fdata())
        beta_map_np = np.reshape(beta_map_np, (1,1,25225))
        betas_tt = torch.Tensor(beta_map_np).to("cpu")
        new_image_pt = torch.from_numpy(images[image_COCO_id])
        reconsTR, clipvoxelsTR = do_reconstructions(betas_tt)
        values_dict = get_top_retrievals(clipvoxelsTR, all_images=all_images, stimulus_trial_counter = stimulus_trial_counter)
        image_array = np.array(reconsTR)[0]
        # If the image has 3 channels (RGB), you need to reorder the dimensions
        if image_array.ndim == 3 and image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))  # Change shape to (height, width, 3)

        # Display the image
        plt.imshow(image_array, cmap='gray' if image_array.ndim == 2 else None)
        plt.axis('off')  # Hide axes
        plt.show()

        # subjInterface.setResultDict allows us to send to the analysis listener immediately
        # subjInterface.setResultDict(name=f'run{run_num}_TR{TR}',
        #                             values=values_dict)
        stimulus_trial_counter += 1
    else:
        if tr_labels_hrf[TR] != "blank":
            values_dict = {}
            image_COCO_id = int(float(tr_labels_hrf[TR].split("_")[1])) - 1
            imsize = 224
            values_dict["ground_truth"] = transforms.Resize((imsize,imsize), antialias=True)(all_images[stimulus_trial_counter]).float().numpy().tolist()
            image_array = np.array(values_dict["ground_truth"])

            # If the image has 3 channels (RGB), you need to reorder the dimensions
            if image_array.ndim == 3 and image_array.shape[0] == 3:
                image_array = np.transpose(image_array, (1, 2, 0))  # Change shape to (height, width, 3)

            # Display the image
            plt.imshow(image_array, cmap='gray' if image_array.ndim == 2 else None)
            plt.axis('off')  # Hide axes
            plt.show()
            # subjInterface.setResultDict(name=f'run{run_num}_TR{TR}',
            #                             values=values_dict)
        else:
            pass
            # when we are not at the end of a stimulus trial, send an empty dictionary to the analysis listener with "pass"
            # subjInterface.setResultDict(name=f'run{run_num}_TR{TR}',
            #                 values={'pass': "pass"})

print(f"==END OF RUN {run_num}!==\n")

