# Re-ACT: Force-Aware Action Chunking with Responsive Temporal Ensemble for Robot Manipulation

### This repo contains the implementation of Re-ACT to operate RB-Y1.
### For real world task, you would also need RB-Y1.

### Repo Structure
- ``imitate_episodes.py`` Train Re-ACT
- ``policy.py`` An adaptor for Re-ACT policy
- ``detr`` Model definitions of Re-ACT, modified from DETR
- ``constants_rby1.py`` Constants for inference with RB-Y1
- ``utils.py`` Utils such as data loading and helper functions
- ``inference_singlearm.py`` Evaluate Re-ACT on single-arm task


### Installation

    conda create -n react python=3.8.10
    conda activate react
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd Re-ACT/detr && pip install -e .
