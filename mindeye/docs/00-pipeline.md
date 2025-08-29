# MindEye Pipeline Overview

This document provides an overview of the real-time MindEye pipeline and describes the stimuli used for the training and real-time sessions. 

## Detailed Pipeline
This is a description of the algorithm that we use to perform real-time image reconstructions and retrievals. We perform real-time analysis on Princeton subject 005 session 06. Our training data consists of subject 005 sessions 01-03 (sessions 04-05 were used for other purposes). See the next section for a detailed description of the image stimuli used in the training and real-time sessions. 

In the quickstart guides: [01-quickstart_simulation.md](01-quickstart_simulation.md) and [02-quickstart_realtime.md](02-quickstart_realtime.md), all of the "prior to real-time" steps below have been completed for you; we provide the preprocessed data from previous scans, fine-tuned MindEye checkpoint, and union mask. 

To run your own real-time analysis, you will additionally need to complete the "prior to real-time" steps; see [03-experiment_guide.md](03-experiment_guide.md).

1. Prior to the real-time session:
    1. MindEye was pre-trained on NSD 7T data from multiple subjects
    2. We collected 3 sessions of 3T data ("MindEye training scans") from subject 005 (sessions 01-03)
    3. Subject 005's data was preprocessed using fMRIPrep; all three sessions were preprocessed together resulting in all functional data in alignment with each other
        * We used the outputs in the subject's native T1 space for all analyses (space-T1w_bold)
    4. Each session's preprocessed data was input to GLMsingle (all 3 sessions together) to obtain single-trial response estimates (betas) for each image-viewing trial
    5. We estimated which voxels responded most reliably to visual images in each session, and then created a "union mask" with the most reliable voxels from each session: 
        * First, we took the NSDgeneral mask in MNI space provided with the NSD dataset and resampled to the subject's native T1w space
        * For each voxel within the subject's NSDgeneral mask, we computed reliability: the correlation of beta values across repeated image presentations (a "reliable voxel" should have consistent responses to the same image presented at different times)
        * We computed the across-session correlation of voxel reliabilities at varying reliability thresholds (i.e., we correlated session 01 and 03 voxel reliabilities and separately correlated session 02 and 03 voxel reliabilities) 
        * For sessions 01 and 02, we independently choose the reliability threshold that maximized the correlation with session 03, resulting in two masks of voxels
        * We took the union of the two masks from the previous step; all subsequent MindEye analyses (fine-tuning the model on sessions 01-03, testing the model in real time on session 06) were limited to this “union mask”
    6. We applied the union mask to the betas from sessions 01-03 to fine-tune MindEye
        * Note: our prior explorations of MindEye showed that limiting the analysis to reliable voxels substantially boosts MindEye performance. In the procedure outlined here, we set out to identify which voxels were reliable based on the training data (sessions 01-03) alone. This way, we could set our mask and conduct the time-consuming process of fine-tuning the model on the training data before the start of real-time testing. 
2. In real-time, stream in the functional data TR-by-TR. 
    1. To be done once at the beginning of the session: Use FLIRT to register the first functional volume of the real-time session to a BOLD reference volume from session 01 in space-T1w
    2. At each TR, motion-correct the new functional volume to the first functional volume of the real-time session and then apply the previously calculated registration
    3. If the current TR corresponds to an image (after accounting for the hemodynamic response function (HRF) delay; assumed here to be on average 7.5s), run a simple GLM (implemented with nilearn) to deconvolve the HRF and produce a single-trial beta and append this to a running list
    4. If the current TR's data should be reconstructed:
        * Z-score voxel-wise over all available betas
            * We use a growing list of betas across all runs, so the mean and standard deviation estimates become more stable as the session progresses
        * Run a forward pass through MindEye to generate retrievals and reconstructions based on the z-scored beta pattern
        * Plot MindEye's outputs

## Stimuli for MindEye training scans
Each MindEye training scan (sessions 01-03) involved viewing 693 images (11 functional runs of 63 images). We split the images into a train and test set of 569 and 124 images, respectively, distributed across all 11 runs. The order of presentation was psuedo-randomized with a constraint to prevent back-to-back repeated presentations of the same image.

There were 531 unique images and 162 repeated images; the repeats were split between the train and test set, described further below. Repeated image presentations (from both the train and test sets; see [here](https://glmsingle.readthedocs.io/en/latest/wiki.html#i-noticed-that-glmsingle-involves-some-internal-cross-validation-is-this-a-problem-for-decoding-style-analyses-where-we-want-to-divide-the-data-into-a-training-set-and-a-test-set)) were used by GLMsingle to perform its optimizations via cross-validation. Repeats were also used to determine reliable voxels, as described above. We fine-tuned the model on the training set of betas from all three sessions simultaneously and used the held-out test set to evaluate our performance. 

### Training set
We use images from NSD, which itself sampled 73,000 images from [Microsoft COCO](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48). Fifty of the training images were shown a total of three times within each session. There were no repeated images from the training set across sessions. 

### Testing set
We use a combination of stimuli from [Wanjia et al. 2021](https://www.nature.com/articles/s41467-021-25126-0#Sec8) and [StyleGAN](http://github.com/NVlabs/stylegan) image variations of stimuli presented to NSD subject 1. We refer to our test set images as "MST pairs"; these are a set of 31 pairs of highly similar images (i.e. lighthouse 1 and lighthouse 2). 13 pairs come from Wanjia et al. 2021 and the other 18 pairs are from StyleGAN. 

The MST images were shown twice each for a total of 124 images in the test set. These images were the same across sessions, and were always held-out during model training.

We use similar pairmates rather than simply choosing a subset of random NSD images to enable further analysis of fine-grained representations. Accordingly, we introduce a retrieval metric, MST 2-alternative forced choice (MST 2-AFC), to evaluate MindEye's ability to discriminate between fMRI responses to these visually and semantically similar images. At an implementation level, this is identical to the retrieval operation, however the model now chooses between the image embeddings of just two pairmates rather than of the pool of all test-set images. 

## Stimuli for real-time scan
The real-time scan (session 06) consisted of 693 images. The first run of 63 images were previously unseen NSD images. These data were preprocessed and betas estimated in real-time, however reconstructions were not attempted. We included this "warm-up run" primarily to get a stable estimate of the mean and standard deviation of each voxel's beta patterns, used for z-scoring new beta patterns passed into MindEye. 

The subsequent runs (2-11) contained pseudo-randomized presentations of the 62 MST images. We required that all 62 images were shown once each before any one was shown for the second time, twice each before any one was shown for the third time, and so on until we filled up all 693 images for the session. We estimated single-trial betas for each image as usual, however for repeated images, we averaged the beta patterns over all available repeats so far. 

The rationale behind this ordering was to first enable evaluation of our pipeline on pure single-trial response estimates (using the first set of repeats) and then to observe potential improvements to our reconstructions obtained by trial averaging. 

Due to the reconstruction lag – waiting for the hemodynamic response to peak, data processing and inference, and displaying the results – we limited the number of attempted reconstructions to 7 evenly spaced trials per run during the real-time session to prevent excessive wait time after the end of each functional run. Additionally, we stopped the session early after 5 functional runs. 
