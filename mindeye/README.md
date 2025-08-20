# Real-time MindEye
Real-time reconstruction of human visual perception from fMRI. 

This is an active, open-source project in collaboration with the [Computational Memory Lab](https://compmem.princeton.edu/) at Princeton University and [Sophont](https://sophontai.com/). We invite anyone interested to explore our open-source code described in this document, reproduce our analyses, and to connect with us on the MedARC Discord server (https://discord.gg/tVR4TWnRM9), an open science research forum operated by Sophont. We are actively seeking collaborators to help extend this framework to new and exciting brain-computer interface applications.

Contact: Rishab Iyer (rsiyer@princeton.edu)

## Overview
We present a first-of-its-kind pipeline that can reconstruct seen images from fMRI brain activity in real-time, using [MindEye2](https://arxiv.org/abs/2403.11207). This serves as a demonstration that RT-Cloud can support state-of-the-art AI workflows for real-time fMRI at any standard MRI facility. By enabling fine-grained decoding of cognitive representations, we substantially increase the utility of real-time fMRI as a brain-computer interface for clinical and scientific applications.

For example, here is a good reconstruction/retrieval from our first-ever real-time session:
  ![alt text](https://github.com/brainiak/rtcloud-projects/raw/main/mindeye/docs/rt-lighthouse-recon.png "Sample real-time reconstruction/retrieval")
The "ground truth" image is what was seen by the subject in the scanner. The reconstruction was generated purely based on the subject's fMRI brain activity just a few seconds later. Retrieval is akin to a multiple choice question for the model: "based on the brain activity, choose the seen image out a pool of *n* images". In this case, the model's choice was correct; its top guess (left) was the true seen image, selected out of a pool of 62. 

Prior work relied on extensive processing of the [Natural Scenes Dataset](https://naturalscenesdataset.org/) (NSD), which collected 30-40 hours of data from each of a few participants. Additionally, NSD used 7 Tesla (7T) MRI which is available at only about 100 sites around the world. Our pipeline involves pre-training MindEye2 on data from NSD and then fine-tuning on just 2-3 hours of 3 Tesla (3T) data from a new participant. After this, the model can support reconstruction of images viewed by that participant using fully real-time-compatible preprocessing.

Going forward, our real-time visual decoding pipeline can potentially support a range of novel applications such as treatment of clinical conditions like depression (e.g., neurofeedback studies where participants are shown how their perception of an image differs from the ground truth image) and it can facilitate probing of fundamental learning mechanisms in the brain. 

## Pipeline
1. Prior to the real-time session:
    1. Pre-train MindEye2 using data from multiple subjects collected using 7T fMRI from NSD
    2. Collect some data from a new participant in 3T fMRI
        1. Select voxels predicted to respond reliably to visual stimuli in a new session
        2. Fine-tune the model using the responses from voxels within this mask
2. In real-time:
    1. Stream DICOM images as the new participant views NSD-like images in 3T fMRI
    2. Perform motion correction and registration to a reference functional volume from a previous session 
    3. Fit a GLM to extract a single-trial response estimate (beta) to each viewed natural scene for each voxel in the pre-defined mask 
    4. Input this beta map into the fine-tuned MindEye2 model to generate image retrievals and/or reconstructions

## Prerequisites
This has been primarily tested on Linux (RHEL and Rocky Linux 9.6). You'll need an internet connection, terminal access, and a GPU. We'll install everything else (Git, Git LFS, Python, Python packages) along the way.

## Setup
In this section, we will set up git, install a uv environment, and clone repositories containing the analysis code, data, and large files.

### Git and Git LFS
1. Check if the Git command line interface (CLI) is installed: `git --version`. If it prints a version number, you're good. If not, install it:
    * MacOS: `brew install git`
    * Ubuntu/Debian Linux: `sudo apt install git`
    * For other systems and additional details, see [documentation](https://docs.github.com/en/get-started/git-basics/set-up-git).

2. We use Git LFS to handle large files such as model weights. Check if it's installed: `git lfs --version`. To install:
    * MacOS: `brew install git-lfs`
    * Ubuntu/Debian Linux: `sudo apt install git-lfs`
    * For other systems and additional details, see [documentation](https://git-lfs.com/). After installation, run **once** per user account: `git lfs install`

### Cloning this repository
1. On the command line, navigate to the directory in which you want to set up this repository: `cd <path/to/directory>`
2. Clone this repository with `git clone https://github.com/brainiak/rtcloud-projects.git`. This will create a folder `rtcloud-projects/`, containing MindEye and other projects.

### Installing uv environment
We use uv, a fast Python package manager ([documentation](https://github.com/astral-sh/uv)) to manage Python versions and dependencies. You may be familiar with tools like conda and pip for package management; uv is a much faster and more modern alternative to these tools. We have exact versions of Python and all dependencies so you can reproduce the environment exactly. Even if you don't have Python installed on your system, uv will take care of this for you.

1. Install uv
    * MacOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    * Verify install using `uv --version`
2. Create the environment
    1. `cd rtcloud-projects/mindeye/config`
    2. Run: `uv sync`. This will: 
        * Install the Python version specified in .python-version
        * Install all dependencies from pyproject.toml pinned by uv.lock
3. Activate the environment
    1. `source .venv/bin/activate`
    2. Check the Python version: `python --version`; it should match the version listed in the file `.python_version` located in the config folder
    3. You can use the command `deactivate` to deactivate the environment

### Installing FSL
We use [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/) for real-time compatible preprocessing, using tools such as MCFLIRT and FLIRT for motion correction and registration, respectively.

1. Install FSL
    * MacOS/Linux: `curl -Ls https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh | sh -s`
    * For other systems and additional details, see [documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index)
    * Successful installation should output "FSL successfully installed"
    * Verify install using `flirt -version`

### Cloning other necessary repositories
1. Clone the repository containing large model files
    * Navigate to the desired location (for example, this can be inside rtcloud-projects/mindeye): `cd <path/to/rtcloud-projects/mindeye>`
    * `git clone https://huggingface.co/datasets/rishab-iyer1/rt_all_data`
2. Clone the repository containing data related to Princeton 3T scans
    * Navigate to the desired location (for example, this can be inside rtcloud-projects/mindeye): `cd /path/to/rtcloud-projects/mindeye`
    * `git clone https://huggingface.co/datasets/rishab-iyer1/3t`
3. Create `config/config.json` with the appropriate file paths. `config/config_example.json` is provided for reference.
    * `project_path` should be set to `<path/to/rtcloud-projects/mindeye>`
    * `storage_path` should be set to `<path/to/rt_all_data>`
    * `data_path` and `derivatives_path` should be set to `<path/to/3t/data>` and `<path/to/3t/derivatives>`, respectively
    * `fsl_path` should be set to the fsl/bin folder containing the executables of various FSL functions. This is often located in your home directory at `~/fsl/bin`
    * The local copy that you update (`config/config.json`) will be automatically ignored by git (due to [.gitignore](.gitignore)). It is good practice to not track user-specific file paths with version control, since each user will have different file systems. 

## Quickstarts
After you have completed the setup instructions above, you can proceed to two quickstart guides: 
1. [`docs/quickstart_simulation.md`](docs/quickstart_simulation.md) to run real-time MindEye in simulation.
    1.  This uses pre-collected data to reproduce the real-time-compatible preprocessing and analysis using a GPU
    2. This is to test real-time MindEye without dependence on real data streaming with RT-Cloud
    3. We recommend starting here before proceeding to the RT-Cloud guide, which requires a more involved setup
2. [`docs/quickstart_realtime.md`](docs/quickstart_realtime.md) to use the full RT-Cloud functionality with MindEye.
    1. This will allow you to perform real-time MindEye analysis with data that is streamed directly from an MRI scanner as it's being collected

## Important Repositories
If you are planning to run your own real-time MindEye scans, here are additional repositories which you may find useful in preparing for and running the real-time session. 
1. [`mindeye_task`](https://github.com/PrincetonCompMemLab/mindeye_task): contains all materials required to run NSD-like MindEye experiments with PsychoPy
2. [`mindeye_preproc`](https://github.com/PrincetonCompMemLab/mindeye_preproc): contains scripts to preprocess offline data 
3. [`mindeye_offline`](https://github.com/PrincetonCompMemLab/mindeye_offline): contains materials to fine-tune MindEye on preprocessed data in preparation for the real-time session
