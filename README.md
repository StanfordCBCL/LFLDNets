# Liquid Fourier Latent Dynamics Networks

This repository contains the code accompanying the paper [1].
We propose an extension of Latent Dynamics Networks (LDNets) [2], namely Liquid Fourier Latent Dynamics Networks (LFLDNets), to create parameterized spatio-temporal surrogate models for multiscale and multiphysics sets of highly nonlinear differential equations on complex geometries.

LFLDNets employ a neurologically-inspired, sparse, liquid neural network for temporal dynamics, relaxing the requirement of a numerical solver for time advancement and accounting for superior performance in terms of tunable parameters, accuracy and efficiency with respect to neural ODEs based on feedforward fully-connected neural networks.
Furthermore, we leverage a Fourier embedding with a tunable kernel for the space coordinates in the reconstruction network to learn complex functions better and faster.

In the framework of computational cardiology, we use LFLDNets to create surrogate models for 3-dimensional anatomies by considering two different test cases arising from cardiac electrophysiology and cardiovascular computational fluid dynamics (`'EP'`, `'CFD'`). Note that a significant amount of RAM is required, especially for the 'CFD' test case.

## Instructions

1. Install a conda environment containing all the required packages:

```bash
conda create -n LFLDNets python=3.8.10 numpy=1.24.3
conda activate LFLDNets
pip install hydra-core
pip install matplotlib
pip install vtk
pip install scipy
pip install dgl
pip install tqdm
pip install meshio
pip install pyvista
pip install torch
pip install torchvision
pip install lightning
pip install tensorflow
pip install ncps
pip install urllib3==1.26.6
pip install "ray[tune]"
pip install optuna
pip install nvidia-tensorrt
```

2. Clone the repository:

```bash
git clone https://github.com/MatteoSalvador/LFLDNets.git
```

3. Activate the conda environment `LFLDNets` by typing `conda activate LFLDNets` (in case it is not already active from the installation procedure at point 1).

4. Download the pickle files `EP.pkl` and `CFD.pkl` at the following [link](https://office365stanford-my.sharepoint.com/:f:/g/personal/msalvad_stanford_edu/EteuMvrj1LZItbTo409xHwQBiYgmP0io1lxqngx48ME8LA). Place them in the `data` folder.

5. **LFLDNets (training):** run the Python script `train.py` with proper settings, specified in the `config.yaml` file. Both single run and distributed hyperparameter tuning with ray are supported. Note that a significant amount of RAM is required, especially for the `'CFD'` test case.

6. **LFLDNets (inference):** run the Python script `test.py` with proper settings, specified in the `config.yaml` file. Note that a significant amount of RAM is required, especially for the `'CFD'` test case.

## Authors

- Matteo Salvador (<msalvad@stanford.edu>)

## References

[1] M. Salvador, A. L. Marsden. [Liquid Fourier Latent Dynamics Networks for fast GPU-based numerical simulations in computational cardiology](http://arxiv.org/abs/2408.09818). *arXiv:2408.09818* (2024).

[2] F. Regazzoni, S. Pagani, M. Salvador, L. Dede', A. Quarteroni. [Learning the intrinsic dynamics of spatio-temporal processes through Latent Dynamics Networks](https://www.nature.com/articles/s41467-024-45323-x). *Nature Communications* (2024).
