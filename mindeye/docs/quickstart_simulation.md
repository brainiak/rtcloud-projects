# Quickstart for Simulated Real-Time MindEye
Quickly set up and run real-time image reconstruction using pre-collected data.

## Introduction
This quickstart guide will walk you through the minimal setup to be able to start producing real-time reconstructions. This guide uses a standalone Jupyter notebook that isolates the real-time MindEye code from the RT-Cloud framework. 

This requires basic familiarity with Python, Git, and the command line. Specific code snippets that you should run will be formatted like ```this text```. Within code snippets, paths that might differ on your computer will be formatted like \<this>.

Use this as a first-pass to set up your environment, download the necessary files, and get familiar with the analysis pipeline. After this, you may want to proceed to [quickstart_realtime.md](quickstart_realtime.md) to use the full RT-Cloud functionality with MindEye.

You have successfully completed this when you are able to run the main analysis loop and it begins generating image reconstructions.

## Prerequisites
This has been primarily tested on Linux (Rocky Linux 9.6). You'll need an internet connection and basic terminal access. We'll install everything else (Git, Git LFS, Python, Python packages) along the way.

## Setting up
In this section, we will install a uv environment and clone repositories containing the analysis code, data, and large files.

### Git and Git LFS
1. Check if the Git command line interface (CLI) is installed: ```git --version```. If it prints a version number, you're good. If not, install it:
    * MacOS: ```brew install git```
    * Ubuntu/Debian Linux: ```sudo apt install git```
    * For other systems and additional details, see [documentation](https://docs.github.com/en/get-started/git-basics/set-up-git).

2. We use Git LFS to handle large files such as model weights. Check if it's installed: ```git lfs --version```. To install:
    * MacOS: ```brew install git-lfs```
    * Ubuntu/Debian Linux: ```sudo apt install git-lfs```
    * For other systems and additional details, see [documentation](https://git-lfs.com/). After installation, run **once** per user account: ```git lfs install```

### Cloning this repository
1. On the command line, navigate to the directory in which you want to set up this repository: ```cd <path/to/directory>```
2. Clone this repository with ```git clone https://github.com/brainiak/rtcloud-projects.git```. This will create a folder ```rtcloud-projects/```, containing MindEye and other projects.

### Installing uv environment
We use [uv](https://github.com/astral-sh/uv) to manage Python versions and dependencies. We have exact versions so you can reproduce the environment exactly.
1. Install uv
    * MacOS/Linux: ```curl -LsSf https://astral.sh/uv/install.sh | sh```
    * Verify install using ```uv --version```
2. Create the environment
    1. ```cd rtcloud-projects/mindeye/config```
    2. Run: ```uv sync```. This will: 
        * Install the Python version specified in .python-version
        * Install all dependencies from pyproject.toml pinned by uv.lock
3. Activate the environment
    1. ```source .venv/bin/activate```
    2. Check the Python version: ```python --version```. It should match the version listed in the file ```.python_version``` located in this folder.


### Cloning other necessary repositories



## Running the notebook in simulated real-time
abc

## Next steps

{% comment %}
Provide a quick recap of what has been accomplished in the quick start as a means of transitioning to next steps. Include 2-3 actionable next steps that the user take after completing the quickstart. Always link to conceptual content on the feature or product. You can also link off to other related information on docs.github.com or in GitHub Skills.
{% endcomment %}
