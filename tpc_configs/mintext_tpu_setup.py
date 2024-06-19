"""
To setup the dependencies of mintext on a TPU pod, fill in the GCP project,
zone, and TPU pod name in the configure_tpc function and run the following
command:

tpc launch mintext_tpu_setup.py
"""

launch_script = r"""#! /bin/bash

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3 \
    tmux \
    htop \
    git \
    nodejs \
    bmon \
    p7zip-full \
    nfs-common


# install miniforge
rm -rf ~/Miniforge3-Linux-x86_64.sh
wget -P ~/ https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ~/Miniforge3-Linux-x86_64.sh -b


cat > $HOME/tpu_environment.yml <<- EndOfFile
name: mintext
channels:
    - pytorch
    - conda-forge
dependencies:
    - python=3.10
    - pip
    - numpy<2
    - scipy
    - matplotlib
    - seaborn
    - jupyter
    - tqdm
    - pytorch=2.3.0
    - cpuonly
    - pip:
        - -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        - jax[tpu]==0.4.28
        - scalax>=0.2.1
        - flax==0.8.3
        - optax==0.2.2
        - transformers==4.41.0
        - torch==2.3.0
        - orbax-checkpoint==0.5.14
        - tensorflow-cpu==2.16.1
        - sentencepiece
        - datasets
        - tpu_pod_commander>=0.1.1
        - mlxu>=0.2.0
        - einops
        - gcsfs
EndOfFile


# install dependencies
source ~/miniconda3/bin/activate
conda init bash
conda env create -f $HOME/tpu_environment.yml
conda activate mintext
"""


configure_tpc(
    project='my-project',
    zone='europe-west4-a',
    name='my-tpu-name',
    launch_script=launch_script,
)