import numpy as np
import pandas as pd
from datetime import datetime, date

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from nilearn.glm.first_level import make_first_level_design_matrix, mean_scaling, run_glm
from nilearn.glm.contrasts import expression_to_contrast_vector, compute_contrast
from subprocess import call

class load_baseline(object):

    def __init__(self,fname):

        self.fname = fname
        self.scores_loaded = False
        
        self.load_scores()

    def load_scores(self):

        try:
            self.score_zero, self.score_std = np.loadtxt(self.fname)
            self.score_loaded = True
            pad_print('Successfully loaded baseline score')

        except:
            pad_print('Loading baseline score failed')

    def check(self):
        if self.score_loaded == False:
            self.load_scores()

def pad_print(msg):
    if len(msg) < 50:
        print('*'+' '*(49-len(msg)) + msg)
    else:
        print(msg)

class tr_timer(object):

    def __init__(self):

        self.zero_time = datetime.now().timestamp()

    def tr_diff(self,TR):

        if TR == 1:
            self.previous_TR = datetime.now().timestamp()
        else:
            self.this_TR = datetime.now().timestamp()
            pad_print(f'time between TRs: {self.this_TR - self.previous_TR:.4f}')
            self.previous_TR = self.this_TR

    def calc_diff(self,start_stop):

        if start_stop == 'start':
            self.calc_start = datetime.now().timestamp()
        elif start_stop == 'stop':
            self.calc_stop = datetime.now().timestamp()
            return self.calc_stop - self.calc_start

def motion_correct(in_=None, reffile=None, out=None):
    # Motion correct to this run's functional reference
    command = f"mcflirt -in {in_} -reffile {reffile} -plots -out {out}"
    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
    pad_print(f"Motion correction time: {B-A:.4f}")

def smooth(fwhm=None, in_=None, out=None):
    # Spatial smoothing
    # full-width half-maximum smoothing kernel (dividing by 2.3548 converts from standard dev. to fwhm)
    command = f'fslmaths {in_} -kernel gauss {fwhm/2.3548} -fmean {out}'
    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
    pad_print(f"Smooth time: {B-A:.4f}")


def ref_mean_scaling(Y, ref, axis=0):
    """Scaling of the data to have percent of baseline change along the
    specified axis

    Parameters
    ----------
    Y : array of shape (n_time_points, n_voxels)
       The input data.

    ref : data used to calculate the mean 

    axis : int, optional
        Axis along which the scaling mean should be calculated. Default=0.

    Returns
    -------
    Y : array of shape (n_time_points, n_voxels),
       The data after mean-scaling, de-meaning and multiplication by 100.
    """
    mean = ref.mean(axis=axis)
    if (mean == 0).any():
        warn('Mean values of 0 observed.'
             'The data have probably been centered.'
             'Scaling might not work as expected')
    mean = np.maximum(mean, 1)
    Y = 100 * (Y / mean - 1)
    return Y


def fast_apply_mask(target=None,mask=None):
    return target[np.where(mask == 1)].T

def fast_glm(Y, events, contrast, confounds=None, seed_ts=None, slice_time_ref=0., 
             t_r=2, drift_model='cosine', high_pass=(1/128), noise_model='ols'):
    '''
    Fast glm estimation
    
    Parameters
    ----------
    
    Y : standardized neural data, type=ndarray, shape=[samples, features]
    
    events : dataframe containing onsets, durations, trial_type, and modulation column
    
    contrast : string containing the desired contrast, accepts same inputs as pd.DataFrame.eval()

    confounds : confound time series, type=ndarray, shape=[samples, features]
    
    seed_ts : if provided, attempts to fit a gPPI using this seed time series
    
    slice_time_ref : value 0-1 for if slice time correction has been applied, default=0 assumes no correction
    
    t_r : tr length in seconds
    
    drift_model : low-pass filtering flavor for GLM
    
    high_pass : cutoff for low-pass filter
    
    noise_model : ols is faster, but ar1 does pre-whitening
    '''
    
    TR = Y.shape[0] #number of samples
    
    #needed for creating the design matrix
    frame_times = np.linspace(slice_time_ref*t_r, (TR-1+slice_time_ref)*t_r, TR)


    #first create the normal design matrix
    dm = make_first_level_design_matrix(frame_times,
                                        events=events, add_regs=confounds,
                                        drift_model=drift_model, high_pass=high_pass)

    #add in the seed time series
    if seed_ts is not None:
        dm['seed'] = seed_ts

        #create the ppi regressors for each trial type
        for con in events.trial_type.unique():
            dm[f'{con}_ppi'] = dm.seed * dm[con]

    labels, estimates = run_glm(Y,dm.values,noise_model=noise_model,n_jobs=1)
    con_map = expression_to_contrast_vector(contrast,dm.columns)
    est = compute_contrast(labels,estimates,con_map,contrast_type='t').effect.mean()
    
    return (est, dm) #returns the effect estimate, and the design matrix which can be saved later