import base64
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import random
import os
import matplotlib.pyplot as plt
import math
import webdataset as wds
import zlib

import json
from PIL import Image
import requests
import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MindEyeModule(nn.Module):
    class RidgeRegression(nn.Module):
        def __init__(self, input_sizes, out_features, seq_len):
            super().__init__()
            self.out_features = out_features
            self.linears = nn.ModuleList(
                [nn.Linear(input_size, out_features) for input_size in input_sizes]
            )
            self.seq_len = seq_len

        def forward(self, x, subj_idx):
            return torch.cat([
                self.linears[subj_idx](x[:, seq]).unsqueeze(1) for seq in range(self.seq_len)
            ], dim=1)

    class BrainNetwork(nn.Module):
        def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=4, drop=.15,
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
                self.mlp(h, h, drop),
            )

        def mixer_block2(self, seq_len, drop):
            return nn.Sequential(
                nn.LayerNorm(seq_len),
                self.mlp(seq_len, seq_len, drop)
            )

        def forward(self, x):
            residual1 = x
            residual2 = x.permute(0, 2, 1)
            for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
                x = block1(x) + residual1
                residual1 = x
                x = x.permute(0, 2, 1)

                x = block2(x) + residual2
                residual2 = x
                x = x.permute(0, 2, 1)

            x = x.reshape(x.size(0), -1)
            backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
            clip = self.clip_proj(backbone)

            dummy = torch.Tensor([[0.], [0.]])
            return backbone, clip, dummy

    def __init__(self, num_voxels, hidden_dim, seq_len, clip_emb_dim, clip_seq_dim,
                 n_blocks, drop=.15):
        super().__init__()
        self.ridge = MindEyeModule.RidgeRegression([
            num_voxels
        ], out_features=hidden_dim, seq_len=seq_len)
        self.backbone = MindEyeModule.BrainNetwork(
            h=hidden_dim,
            in_dim=hidden_dim,
            out_dim=clip_emb_dim * clip_seq_dim,
            seq_len=seq_len,
            n_blocks=n_blocks,
            drop=drop,
            clip_size=clip_emb_dim
        )
        self.diffusion_prior = None

    def forward(self, x):
        return x

    def build_diffusion_prior(self, clip_emb_dim, clip_seq_dim, depth, dim_head, heads,
                               timesteps, cond_drop_prob=0.2, image_embed_scale=None):
        from models import BrainDiffusionPrior, PriorNetwork

        prior_network = PriorNetwork(
            dim=clip_emb_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens=clip_seq_dim,
            learned_query_mode="pos_emb"
        )

        self.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=clip_emb_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=cond_drop_prob,
            image_embed_scale=image_embed_scale,
        )
        return self.diffusion_prior

def compress_and_encode_image(image_array):
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    compressed_data = zlib.compress(image_array.tobytes())
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    return encoded_data

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def batchwise_pearson_correlation(Z, B):
    # Calculate means
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)

    # Subtract means
    Z_centered = Z - Z_mean
    B_centered = B - B_mean

    # Calculate Pearson correlation coefficient
    numerator = Z_centered @ B_centered.T
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T

    pearson_correlation = (numerator / denominator)
    return pearson_correlation

def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def prenormed_batchwise_cosine_similarity(Z,B):
    return (Z @ B.T).T

def cosine_similarity(Z,B,l=0):
    Z = nn.functional.normalize(Z, p=2, dim=1)
    B = nn.functional.normalize(B, p=2, dim=1)
    # if l>0, use distribution normalization
    # https://twitter.com/YifeiZhou02/status/1716513495087472880
    Z = Z - l * torch.mean(Z,dim=0)
    B = B - l * torch.mean(B,dim=0)
    cosine_similarity = (Z @ B.T).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def get_non_diagonals(a):
    a = torch.triu(a,diagonal=1)+torch.tril(a,diagonal=-1)
    # make diagonals -1
    a=a.fill_diagonal_(-1)
    return a

def gather_features(image_features, voxel_features, accelerator):  
    all_image_features = accelerator.gather(image_features.contiguous())
    if voxel_features is not None:
        all_voxel_features = accelerator.gather(voxel_features.contiguous())
        return all_image_features, all_voxel_features
    return all_image_features

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def soft_siglip_loss(preds, targs, temp, bias):
    temp = torch.exp(temp)
    
    logits = (preds @ targs.T) * temp + bias
    # diagonals (aka paired samples) should be >0 and off-diagonals <0
    labels = (targs @ targs.T) - 1 + (torch.eye(len(targs)).to(targs.dtype).to(targs.device))

    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels[:len(preds)])) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels[:,:len(preds)])) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco_hard_siglip_loss(preds, targs, temp, bias, perm, betas):
    temp = torch.exp(temp)
    
    probs = torch.diag(betas)
    probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

    logits = (preds @ targs.T) * temp + bias
    labels = probs * 2 - 1
    #labels = torch.eye(len(targs)).to(targs.dtype).to(targs.device) * 2 - 1
    
    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels)) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels)) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_clip_target(clip_target, perm, select, betas):
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = clip_target[select] * betas[select].reshape(-1, 1) + \
        clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    return clip_target

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss
    
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))
    return trainable
    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def resize(img, img_size=128):
    if img.ndim == 3: img = img[None]
    return nn.functional.interpolate(img, size=(img_size, img_size), mode='nearest')

pixcorr_preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])
def pixcorr(images,brains,nan=True):
    all_images_flattened = pixcorr_preprocess(images).reshape(len(images), -1)
    all_brain_recons_flattened = pixcorr_preprocess(brains).reshape(len(brains), -1)
    if nan:
        corrmean = torch.nanmean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    else:
        corrmean = torch.mean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    return corrmean

def select_annotations(annots, random=True):
    """
    There are 5 annotations per image. Select one of them for each image.
    """
    for i, b in enumerate(annots):
        t = ''
        if random:
            # select random non-empty annotation
            while t == '':
                rand = torch.randint(5, (1,1))[0][0]
                t = b[rand]
        else:
            # select first non-empty annotation
            for j in range(5):
                if b[j] != '':
                    t = b[j]
                    break
        if i == 0:
            txt = np.array(t)
        else:
            txt = np.vstack((txt, t))
    txt = txt.flatten()
    return txt

def do_reconstructions(model, betas_tt, diffusion_engine, vector_suffix, imsize, device,
                       num_samples_per_image=1):
    print('starting reconstruction!')
    model.to(device)
    model.eval().requires_grad_(False)
    if model.diffusion_prior is None:
        raise ValueError("MindEyeModule.diffusion_prior is not initialized. Call build_diffusion_prior first.")
    clipvoxelsTR = None
    reconsTR = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        voxel = betas_tt.to(device)
        voxel_ridge = model.ridge(voxel[:, [0]], 0)
        backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
        clip_voxels = clip_voxels0
        backbone = backbone0
        blurry_image_enc = blurry_image_enc0[0]
        clipvoxelsTR = clip_voxels.cpu()
        prior_out = model.diffusion_prior.p_sample_loop(
            backbone.shape,
            text_cond=dict(text_embed=backbone),
            cond_scale=1.,
            timesteps=20
        )
        for i in range(len(voxel)):
            samples = unclip_recon(
                prior_out[[i]],
                diffusion_engine,
                vector_suffix,
                num_samples=num_samples_per_image
            )
            if reconsTR is None:
                reconsTR = samples.cpu()
            else:
                reconsTR = torch.vstack((reconsTR, samples.cpu()))

            reconsTR = transforms.Resize((imsize, imsize), antialias=True)(reconsTR)

    return reconsTR, clipvoxelsTR

def do_retrievals(clip_img_embedder, clipvoxel, all_images, imsize, device,
                  total_retrievals=1):
    values_dict = {}
    with torch.amp.autocast('cuda', dtype=torch.float16):
        emb = clip_img_embedder(
            torch.reshape(all_images, (all_images.shape[0], 3, imsize, imsize)).to(device)
        ).float()
        emb = emb.cpu()
        emb_ = clipvoxel
        emb = emb.reshape(len(emb), -1)
        emb_ = np.reshape(emb_, (1, 425984))
        emb = nn.functional.normalize(emb, dim=-1)
        emb_ = nn.functional.normalize(emb_, dim=-1)
        emb_ = emb_.float()
        fwd_sim = batchwise_cosine_similarity(emb_, emb)
        print("Given Brain embedding, find correct Image embedding")

    fwd_sim = np.array(fwd_sim.cpu())
    which = np.flip(np.argsort(fwd_sim, axis=0))

    for attempt in range(total_retrievals):
        image_tensor = all_images[which[attempt].copy()]
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        resized = transforms.Resize((imsize, imsize), antialias=True)(image_tensor.unsqueeze(0))
        image_array = (resized.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
        values_dict[f"attempt{(attempt+1)}"] = compress_and_encode_image(image_array)

    return values_dict


def _to_flat_float(x):
    """Flatten an array-like / torch.Tensor to a 1-D float32 CPU tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).reshape(-1).cpu()
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(-1)


def cpd_from_embeddings(pred, correct, foil, normalize=True):
    """Cortical Pairmate Distinctiveness from already-computed CLIP embeddings.

    Defines the axis in CLIP space spanned by the ``correct`` image and its
    pairmate ``foil``, then projects the brain-predicted embedding ``pred`` onto
    that axis. Returns a signed, unbounded scalar:

        +1  -> exactly at the correct image's embedding
        -1  -> exactly at the foil's embedding
         0  -> the midpoint between them

    cpd = 2 * <pred - midpoint, correct - foil> / ||correct - foil||^2

    All inputs are flattened. With ``normalize=True`` each flattened vector is
    L2-normalized first, matching the normalized CLIP space that
    :func:`do_retrievals` / :func:`calculate_retrieval_metrics` use.
    """
    pred = _to_flat_float(pred)
    correct = _to_flat_float(correct)
    foil = _to_flat_float(foil)
    if normalize:
        pred = F.normalize(pred, dim=-1)
        correct = F.normalize(correct, dim=-1)
        foil = F.normalize(foil, dim=-1)
    axis = correct - foil
    midpoint = (correct + foil) / 2.0
    denom = torch.dot(axis, axis)
    cpd = 2.0 * torch.dot(pred - midpoint, axis) / denom
    return float(cpd)


def compute_cpd(clip_img_embedder, clipvoxel, correct_image, foil_image,
                imsize, device, normalize=True):
    """Cortical Pairmate Distinctiveness for one trial.

    Mirrors :func:`do_retrievals`: embeds the ``correct_image`` and its pairmate
    ``foil_image`` through the CLIP image embedder, then projects the
    brain-predicted CLIP embedding ``clipvoxel`` onto the correct-vs-foil axis
    (see :func:`cpd_from_embeddings`).

    correct_image / foil_image : image tensors, reshaped to (1, 3, imsize, imsize).
    clipvoxel                  : brain-predicted CLIP embedding (any shape; flattened).
    """
    use_amp = str(device).startswith("cuda")

    def _embed(img):
        img = img.reshape(1, 3, imsize, imsize).to(device)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                emb = clip_img_embedder(img)
        else:
            emb = clip_img_embedder(img)
        return emb.float().cpu()

    with torch.no_grad():
        correct_emb = _embed(correct_image)
        foil_emb = _embed(foil_image)

    return cpd_from_embeddings(clipvoxel, correct_emb, foil_emb, normalize=normalize)


def benchmark_reconstruction(model, diffusion_engine, vector_suffix, device, imsize,
                             iterations=5, warmup=2, batch_size=1):
    from time import perf_counter

    in_features = model.ridge.linears[0].in_features
    seq_len = model.ridge.seq_len
    betas = torch.randn(batch_size, seq_len, in_features)

    # Warm-up
    for _ in range(max(warmup, 0)):
        do_reconstructions(model, betas.clone(), diffusion_engine, vector_suffix, imsize, device)

    timings = []
    for _ in range(iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = perf_counter()
        do_reconstructions(model, betas.clone(), diffusion_engine, vector_suffix, imsize, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timings.append(perf_counter() - start)

    return {
        "mean": float(np.mean(timings)),
        "std": float(np.std(timings)),
        "iterations": iterations,
        "warmup": warmup,
    }

from generative_models.sgm.util import append_dims
def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04):
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img

        # clip_img_tokenized = clip_img_embedder(image) 
        # tokens = clip_img_tokenized
        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples

#  Numpy Utility 
def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        
# Torch fwRF
def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125):
    teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T)/temp
    teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T)/temp
    student_teacher_aug = (student_preds @ teacher_aug_preds.T)/temp
    student_teacher_aug_t = (teacher_aug_preds @ student_preds.T)/temp

    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def ddp(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def process_design(filename):
    """
    Process design CSV file and extract relevant trial information
    
    Args:
        filename: Path to CSV file containing design information
    
    Returns:
        data: Pandas DataFrame containing all design information
        starts: Array of trial start times
        images: Array of image names
        is_new_run: Array of boolean flags indicating new runs
        image_names: Array of image names
    """
    import pandas as pd
    data = pd.read_csv(filename)
    data = data.dropna(subset=['current_image'])  # there are nans (blank cells) between runs
    starts = data['trial.started'].values[10:]
    images = data['current_image'].values[10:]
    is_new_run = data['is_new_run'].values[10:]
    image_names = data['current_image'].values[10:]
    return data, starts, images, is_new_run, image_names


def load_design_files(sub, session, func_task_name, designdir, design_ses_list=None):
    """
    Load design files for a given subject and session
    
    Args:
        sub: str, subject ID (e.g., 'sub-001')
        session: str, session ID (e.g., 'ses-01' or 'all')
        func_task_name: str, task name (e.g., 'A', 'B', 'C')
        designdir: str, path to design directory
        design_ses_list: list, list of sessions to process when session='all'
    
    Returns:
        tuple containing:
        - data: pandas DataFrame with design information
        - starts: array of trial start times
        - images: array of image names
        - is_new_run: array of boolean flags for new runs
        - image_names: array of image names
        - unique_images: array of unique image names
        - len_unique_images: int, number of unique images
    """
    import pandas as pd
    if (sub=='sub-001' and session=='ses-01') or (sub=='sub-002' and session=='ses-01'):
        filename = f"{designdir}/csv/{sub}_{session}.csv"
        data = pd.read_csv(filename)
        images = data['current_image'].values[23:]
        starts = data['trial.started'].values[23:]
        is_new_run = data['is_new_run'].values[23:]
        image_names = data['current_image'].values[23:]
        
    elif (sub=='sub-001' and session in ('ses-02', 'ses-03', 'ses-04', 'ses-05')) or \
         (sub=='sub-002' and session in ('ses-02')) or sub=='sub-003' or \
         (sub=='sub-004' and session in ('ses-01', 'ses-02')) or \
         (sub=='sub-005' and session in ('ses-01', 'ses-02', 'ses-03', 'ses-06', 'ses-07', 'ses-08')) or \
         (sub=='sub-006' and session in ('ses-01')):
        
        if (sub=='sub-001' and session in ('ses-05')):
            if func_task_name == 'A':
                filename = f"{designdir}/csv/{sub}_ses-05.csv"
            elif func_task_name == 'B':
                filename = f"{designdir}/csv/{sub}_ses-06.csv"
            elif func_task_name == 'C':
                filename = f"{designdir}/csv/{sub}_ses-07.csv"

        elif (sub=='sub-002' and session in ('ses-02')):
            if func_task_name == 'A':
                filename = f"{designdir}/csv/{sub}_ses-06.csv"
            elif func_task_name == 'B':
                filename = f"{designdir}/csv/{sub}_ses-07.csv"
            elif func_task_name == 'C':
                filename = f"{designdir}/csv/{sub}_ses-05.csv"
        
        elif (sub=='sub-004' and session in ('ses-01')):
            if func_task_name == 'A':
                filename = f"{designdir}/csv/{sub}_ses-07.csv"
            elif func_task_name == 'B':
                filename = f"{designdir}/csv/{sub}_ses-05.csv"
            elif func_task_name == 'C':
                filename = f"{designdir}/csv/{sub}_ses-06.csv"
        elif (sub=='sub-004' and session in ('ses-02')):
            assert func_task_name == 'C'
            filename = f"{designdir}/csv/{sub}_ses-08.csv"

        elif (sub=='sub-005' and session in ('ses-01', 'ses-02', 'ses-03', 'ses-06', 'ses-07', 'ses-08')) or sub=='sub-006' and session in ('ses-01'):
            filename = f"{designdir}/csv/{sub}_{session}.csv"
        
        data, starts, images, is_new_run, image_names = process_design(filename)
        print(f"Data shape: {data.shape}")

    elif sub in ('sub-001', 'sub-004', 'sub-005') and session == 'all':
        assert design_ses_list is not None, "design_ses_list must be provided when session='all'"

        data_list = []
        starts_list = []
        images_list = []
        is_new_run_list = []
        image_names_list = []

        for ses in design_ses_list:
            filename = f"{designdir}/csv/{sub}_{ses}.csv"
            print(f"Loading: {filename}")

            data_tmp, starts_tmp, images_tmp, is_new_run_tmp, image_names_tmp = process_design(filename)

            data_list.append(data_tmp)
            starts_list.append(starts_tmp)
            images_list.append(images_tmp)
            is_new_run_list.append(is_new_run_tmp)
            image_names_list.append(image_names_tmp)
        # Concatenate all lists
        data = pd.concat(data_list, ignore_index=True)
        starts = np.concatenate(starts_list)
        images = np.concatenate(images_list)
        is_new_run = np.concatenate(is_new_run_list)
        image_names = np.concatenate(image_names_list)
    else:
        raise Exception("undefined subject and/or session")

    print(f"Using design file: {filename}")
    
    unique_images = np.unique(images.astype(str))
    len_unique_images = len(unique_images)
    print('Total number of images:', len(images))
    print("Number of unique images:", len_unique_images)
    
    return data, starts, images, is_new_run, image_names, unique_images, len_unique_images


def create_design_matrix(images, starts, is_new_run, unique_images, n_runs, n_trs, len_unique_images, interval):
    """
    Create a design matrix for GLMsingle analysis
    
    Args:
        images: List of image filenames
        starts: List of trial start times
        is_new_run: List of boolean flags indicating if trial is last in run
        unique_images: List of unique image filenames
        n_runs: Number of runs
        n_trs: Number of TRs per run
        len_unique_images: Length of unique_images list
    
    Returns:
        design: List of design matrices, one per run
    """
    design = [np.zeros((n_trs, len_unique_images)) for _ in range(n_runs)]
    starting_time = starts[0]
    cur_run = 0
    first_trial_of_new_run = False
    
    for i in range(len(images)):
        if first_trial_of_new_run: # is the first trial of a new run?
            starting_time = starts[i]
            cur_run += 1
            first_trial_of_new_run = False
            
        if is_new_run[i] == 1: # is this the last trial of the run?
            first_trial_of_new_run = True
            if n_runs == 1:
                break
        
        if images[i] == "blank.jpg":
            continue
    
        image_idx = np.where(str(images[i])==unique_images)[0].item()
        timepoint = int(np.round((starts[i] - starting_time) / interval))
        
        design[cur_run][timepoint, image_idx] = 1
            
    return design


#########################################################
####### Evaluation utils
#########################################################

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from scipy import stats
from tqdm import tqdm
def calculate_retrieval_metrics(all_clip_voxels, all_images):
    print("Loading clip_img_embedder")
    try:
        print(clip_img_embedder)
    except:
        clip_img_embedder = FrozenOpenCLIPImageEmbedder(
            arch="ViT-bigG-14",
            version="laion2b_s39b_b160k",
            output_tokens=True,
            only_tokens=True,
        )
        clip_img_embedder.to(device)
    clip_seq_dim = 256
    clip_emb_dim = 1664

    all_fwd_acc = []
    all_bwd_acc = []

    assert len(all_images) == len(all_clip_voxels)  
    print("The total pool of images and clip voxels to do retrieval on is: ", len(all_images))
    all_percent_correct_fwds, all_percent_correct_bwds = [], []

    with torch.cuda.amp.autocast(dtype=torch.float16):
        print("Creating embeddings for images")
        with torch.no_grad():
            all_emb = clip_img_embedder(all_images.to(torch.float16).to(device)).float() # CLIP-Image

        all_emb_ = all_clip_voxels # CLIP-Brain

        print("Calculating retrieval metrics")
        # flatten if necessary
        all_emb = all_emb.reshape(len(all_emb),-1).to(device)
        all_emb_ = all_emb_.reshape(len(all_emb_),-1).to(device)

        # l2norm 
        all_emb = nn.functional.normalize(all_emb,dim=-1)
        all_emb_ = nn.functional.normalize(all_emb_,dim=-1)

        all_labels = torch.arange(len(all_emb)).to(device)
        all_bwd_sim = batchwise_cosine_similarity(all_emb,all_emb_)  # clip, brain
        all_fwd_sim = batchwise_cosine_similarity(all_emb_,all_emb)  # brain, clip

        # if "ses-0" not in model_name or "ses-01" in model_name or "ses-04" in model_name:
        #     assert len(all_fwd_sim) == 100
        #     assert len(all_bwd_sim) == 100
        # else:
        #     assert len(all_fwd_sim) == 50
        #     assert len(all_bwd_sim) == 50
        
        all_percent_correct_fwds = topk(all_fwd_sim, all_labels, k=1).item()
        all_percent_correct_bwds = topk(all_bwd_sim, all_labels, k=1).item()

    all_fwd_acc.append(all_percent_correct_fwds)
    all_bwd_acc.append(all_percent_correct_bwds)

    all_fwd_sim = np.array(all_fwd_sim.cpu())
    all_bwd_sim = np.array(all_bwd_sim.cpu())

    print(f"overall fwd percent_correct: {all_fwd_acc[0]:.4f}")
    print(f"overall bwd percent_correct: {all_bwd_acc[0]:.4f}")

    return all_fwd_acc[0], all_bwd_acc[0]


from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1
    
def calculate_pixcorr(all_recons, all_images):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess(all_recons).reshape(len(all_recons), -1).cpu()

    print(all_images_flattened.shape)
    print(all_recons_flattened.shape)

    corr_stack = []

    corrsum = 0
    for i in tqdm(range(len(all_images))):
        corrcoef = np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1]
        if np.isnan(corrcoef):
            print("WARNING: CORRCOEF WAS NAN")
            corrcoef = 0
        corrsum += corrcoef
        corr_stack.append(corrcoef)
    corrmean = corrsum / len(all_images)

    pixcorr = corrmean
    print(f"Pixel Correlation: {pixcorr}")
    return pixcorr

from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(all_recons, all_images):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])

    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess(all_recons).permute((0,2,3,1)).cpu())
    print("converted, now calculating ssim...")

    ssim_score=[]
    for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
        ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

    ssim_ = np.mean(ssim_score)
    print(f"SSIM: {ssim_}")
    return ssim_

from torchvision.models import alexnet, AlexNet_Weights
def calculate_alexnet(all_recons, all_images, layers = [2, 5]):
    print("Loading AlexNet")
    alex_weights = AlexNet_Weights.DEFAULT
    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
    alex_model.eval().requires_grad_(False).to(device)

    # see alex_weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    alexnet2 = None
    alexnet5 = None
    if 2 in layers:
        layer = 'early, AlexNet(2)'
        print(f"\n---{layer}---")
        all_per_correct = two_way_identification(all_recons, all_images, 
                                                            alex_model, preprocess, 'features.4')
        alexnet2 = np.mean(all_per_correct)
        print(f"2-way Percent Correct (early AlexNet): {alexnet2:.4f}")

    if 5 in layers:
        layer = 'mid, AlexNet(5)'
        print(f"\n---{layer}---")
        all_per_correct = two_way_identification(all_recons, all_images, 
                                                            alex_model, preprocess, 'features.11')
        alexnet5 = np.mean(all_per_correct)
        print(f"2-way Percent Correct (mid AlexNet): {alexnet5:.4f}")

    return alexnet2, alexnet5


from torchvision.models import inception_v3, Inception_V3_Weights
def calculate_inception_v3(all_recons, all_images):
    print("Loading Inception V3")
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                            return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False).to(device)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    all_per_correct = two_way_identification(all_recons, all_images,
                                            inception_model, preprocess, 'avgpool')
            
    inception = np.mean(all_per_correct)
    print(f"2-way Percent Correct (Inception V3): {inception:.4f}")

    return inception


import clip as clip_torch
def calculate_clip(all_recons, all_images):
    print("Loading CLIP")
    clip_model, preprocess = clip_torch.load("ViT-L/14", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    all_per_correct = two_way_identification(all_recons, all_images,
                                            clip_model.encode_image, preprocess, None) # final layer
    clip_ = np.mean(all_per_correct)
    print(f"2-way Percent Correct (CLIP): {clip_:.4f}")

    return clip_

import scipy as sp
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
def calculate_efficientnet_b1(all_recons, all_images):
    print("Loading EfficientNet B1")
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                        return_nodes=['avgpool'])
    eff_model.eval().requires_grad_(False).to(device)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = eff_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    effnet_nomean = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))])
    effnet = effnet_nomean.mean()
    print("Distance EfficientNet B1:",effnet)

    return effnet

def calculate_swav(all_recons, all_images):
    print("Loading SwAV")
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, 
                                        return_nodes=['avgpool'])
    swav_model.eval().requires_grad_(False).to(device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = swav_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    swav_nomean = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))])
    swav = swav_nomean.mean()
    print("Distance SwAV:",swav)

    return swav

def deduplicate_tensors(all_clipvoxels_save_tensor, all_ground_truth_save_tensor):
    """
    Remove duplicate images in the ground truth and average the corresponding voxel embeddings.
    
    Arguments:
        all_clipvoxels_save_tensor (torch.Tensor): tensor of shape (N, 1, 256, 1664) containing voxel predictions.
        all_ground_truth_save_tensor (torch.Tensor): tensor of shape (N, 3, 224, 224) containing images.
        
    Returns:
        new_voxels (torch.Tensor): deduplicated voxels tensor with averaged embeddings, shape (M, 1, 256, 1664),
                                   where M is the number of unique images.
        new_ground_truth (torch.Tensor): deduplicated ground truth images tensor, shape (M, 3, 224, 224).
        duplicate_pairs (list of tuples): list of tuples (i, j) where image at index j was found to be 
                                          a duplicate of image at index i.
    """
    # Number of samples
    num_samples = all_ground_truth_save_tensor.shape[0]
    
    unique_indices = []  # Stores the indices of first occurrences of unique images
    duplicate_groups = {}  # Map each unique index -> list of all indices (including the primary) that are identical
    duplicate_pairs = []  # List to store (primary, duplicate) pairs

    # Loop over all images in the ground truth tensor
    for i in range(num_samples):
        current_img = all_ground_truth_save_tensor[i]
        # flag to detect whether the current image has a duplicate already
        found_duplicate = False
        for unique_idx in unique_indices:
            if torch.equal(current_img, all_ground_truth_save_tensor[unique_idx]):
                # Duplicate found: record the pair and add current index to the duplicate group
                duplicate_pairs.append((unique_idx, i))
                duplicate_groups[unique_idx].append(i)
                found_duplicate = True
                break
        if not found_duplicate:
            # New image; mark as unique and initialize its duplicate group list
            unique_indices.append(i)
            duplicate_groups[i] = [i]
    
    # Build new tensors
    dedup_ground_truth_list = []
    dedup_voxels_list = []
    
    for unique_idx in unique_indices:
        # For ground truth, we keep the primary image (the first occurrence)
        dedup_ground_truth_list.append(all_ground_truth_save_tensor[unique_idx])
        
        # For voxels, average the voxels over all indices in this duplicate group
        indices = duplicate_groups[unique_idx]
        voxels_group = all_clipvoxels_save_tensor[indices]  # shape: (n_group, 1, 256, 1664)
        averaged_voxels = voxels_group.mean(dim=0)
        dedup_voxels_list.append(averaged_voxels)
    
    # Stack lists into tensors along the first dimension
    new_ground_truth = torch.stack(dedup_ground_truth_list, dim=0)
    new_voxels = torch.stack(dedup_voxels_list, dim=0)
    
    return new_voxels, new_ground_truth, duplicate_pairs


def find_paired_indices(x):
    unique_elements, counts = torch.unique(x, return_counts=True)
    repeated_elements = unique_elements[counts > 1]
    paired_indices = []
    
    for element in repeated_elements:
        indices = (x == element).nonzero(as_tuple=True)[0]
        # Instead of creating pairs, just collect the entire set of indices once
        paired_indices.append(indices[:len(indices)].tolist())
    
    return paired_indices


def process_images(image_names, unique_images, remove_close_to_MST=False, remove_random_n=False, imgs_to_remove=None, sub=None, session=None):
    import re
    image_idx = np.array([])
    vox_image_names = np.array([])
    all_MST_images = {}
    
    for i, im in enumerate(image_names):
        if im == "blank.jpg" or str(im) == "nan":
            continue
                
        if remove_close_to_MST and "closest_pairs" in im:
            continue
        
        if remove_random_n and im in imgs_to_remove:
            continue
            
        vox_image_names = np.append(vox_image_names, im)
        image_idx_ = np.where(im == unique_images)[0].item()
        image_idx = np.append(image_idx, image_idx_)
        
        if sub == 'ses-01' and session in ('ses-01', 'ses-04'):
            if ('w_' in im or 'paired_image_' in im or re.match(r'all_stimuli/rtmindeye_stimuli/\d{1,2}_\d{1,3}\.png$', im) 
                or re.match(r'images/\d{1,2}_\d{1,3}\.png$', im)):
                all_MST_images[i] = im
        elif 'MST' in im:
            all_MST_images[i] = im
    
    image_idx = torch.Tensor(image_idx).long()
    unique_MST_images = np.unique(list(all_MST_images.values()))
    
    MST_ID = np.array([], dtype=int)
    if remove_close_to_MST:
        close_to_MST_idx = np.array([], dtype=int)
    if remove_random_n:
        random_n_idx = np.array([], dtype=int)
    
    vox_idx = np.array([], dtype=int)
    j = 0  # Counter for indexing vox based on removed images
    
    for i, im in enumerate(image_names):
        if im == "blank.jpg" or str(im) == "nan":
            continue
        
        if remove_close_to_MST and "closest_pairs" in im:
            close_to_MST_idx = np.append(close_to_MST_idx, i)
            continue
        
        if remove_random_n and im in imgs_to_remove:
            vox_idx = np.append(vox_idx, j)
            j += 1
            continue
        
        j += 1
        curr = np.where(im == unique_MST_images)
        
        if curr[0].size == 0:
            MST_ID = np.append(MST_ID, len(unique_MST_images))  # Out of range index for filtering later
        else:
            MST_ID = np.append(MST_ID, curr)
    
    assert len(MST_ID) == len(image_idx)
    
    pairs = find_paired_indices(image_idx)
    pairs = sorted(pairs, key=lambda x: x[0])
    
    return image_idx, vox_image_names, pairs


def zscore(data,train_mean=None,train_std=None):
    # assuming that first dim is num_samples and second dim is num_voxels
    if train_mean is None:
        train_mean = np.mean(data,axis=0)
    if train_std is None:
        train_std = np.std(data,axis=0)
    zscored_data = (data - train_mean) / (train_std + 1e-6)
    return zscored_data


def vectorized_pearsonr(X, Y):
    """
    Calculates Pearson correlation coefficients and p-values for multiple pairs of arrays.

    Args:
        X (numpy.ndarray): A 2D array where each row represents a variable.
        Y (numpy.ndarray): A 2D array where each row represents a variable.
                           Must have the same number of columns as X.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - Pearson correlation coefficients (r values).
            - p-values.
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of columns")

    X_mean = np.mean(X, axis=0, keepdims=True)
    Y_mean = np.mean(Y, axis=0, keepdims=True)

    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    numerator = np.sum(X_centered * Y_centered, axis=1)

    denominator = np.sqrt(np.sum(X_centered**2, axis=1) * np.sum(Y_centered**2, axis=1))

    r = numerator / denominator
    
    n = X.shape[1]
    
    #handle potential division by zero
    r = np.where(denominator == 0, 0, r)

    # Calculate p-values using the t-distribution
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
    
    # Handle cases where r is NaN due to division by zero
    p = np.where(np.isnan(p), 1.0, p)

    return r, p