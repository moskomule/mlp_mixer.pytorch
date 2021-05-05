# mlp_mixer.pytorch

PyTorch implementation
of [Tolstikhin et al. 2021 "MLP-Mixer: An all-MLP Architecture for Vision"](https://arxiv.org/abs/2105.01601).

## Requirements

```commandline
conda create -n mixer python=3.9
conda activate mixer
conda install -c pytorch -c conda-forge pytorch torchvision cudatoolkit=11.1
pip install -U homura-core chika rich
```

## ImageNet classification

```
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUs} main.py ...
```