"""
To launch llama3 finetuning with mintext on a TPU pod, fill in the GCP project,
zone, and TPU pod name in the configure_tpc function and run the following
command:

tpc launch mintext_tpu_llama3_finetune.py
"""

from pathlib import Path

launch_script = r"""#! /bin/bash

source ~/miniforge3/bin/activate mintext

export PYTHONPATH="$PYTHONPATH:$HOME/mintext"
export WANDB_API_KEY='<your_wandb_api_key>'
export HF_TOKEN='<your huggingface access token>'
export experiment_name='llama3_8b_finetune'



python -m mintext.train \
    --mesh_dim='1,-1,1' \
    --total_steps=10000 \
    --optimizer.lr=3e-6 \
    --optimizer.end_lr=3e-7 \
    --optimizer.lr_warmup_steps=1000 \
    --optimizer.lr_decay_steps=10000 \
    --optimizer.weight_decay=0.03 \
    --log_freq=100 \
    --dtype='bf16' \
    --param_dtype='fp32' \
    --save_model_freq=1000 \
    --save_milestone_freq=5000 \
    --tokenizer='meta-llama/Meta-Llama-3-8B' \
    --load_params_checkpoint='gs://path/to/your/converted/llama3_8b/checkpoint' \
    --llama.base_model='llama3_8b' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.path='gs://path/to/your/dataset' \
    --train_dataset.seq_length=2048 \
    --train_dataset.batch_size=256 \
    --train_dataset.tokenizer_processes=8 \
    --logger.online=True \
    --logger.project="$experiment_name" \
    --logger.output_dir="gs://young-aidm-euw4/experiment_output/young/mintext/$experiment_name" \
    --logger.wandb_dir="$HOME/experiment_output/$experiment_name" \
    --logger.notes="$experiment_name"


read
"""

mintext_path = Path(__file__).parent.parent.as_posix()

# Fill in your TPU information here
configure_tpc( # type: ignore
    project='<your GCP project>',
    zone='<your TPC zone>',
    name='<your TPU pod name>',
    accelerator_type='v5litepod-128',
    runtime_version='v2-alpha-tpuv5-lite',
    reserved=True,
    launch_script=launch_script,
    upload_path=f'{mintext_path}:~/mintext',
)
