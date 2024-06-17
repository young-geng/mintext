# mintext
Mintext is a minimal but scalable implementation of large language models in JAX.
Specifically, it implements the LLaMA architecture in a clean and modular way,
which makes it easy to modify and extend. The codebase is designed to be a
didactic example of how one can implement a large language model from scratch
in JAX with fairly minimal code, while still retaining the ability to scale to
large models on thousands of accelerators.

## Installation
Mintext uses conda to manage dependencies, and the installation process differs
depending on whether you are using GPUs or a TPUs.


### GPU Installation
To install mintext with GPU support, simply run the following command:
```bash
git clone https://github.com/young-geng/mintext.git
cd mintext
conda env create -f environment_gpu.yml
conda activate mintext
export PYTHONPATH="${PWD}:$PYTHONPATH"
```


### TPU Installation
The recommended way to install mintext on TPUs is to use the `tpu_pod_commander`
package. First follow the instructions in the [TPU Pod Commander README](
https://github.com/young-geng/tpu_pod_commander) to install the package.
Then, install mintext by running the following commands on your local machine:
```bash
git clone https://github.com/young-geng/mintext.git
cd mintext
tpc launch tpc_configs/mintext_tpu_setup.py \
    --name=your-tpu-pod-name \
    --project=your-gcp-project \
    --zone=your-zone \
```


