#!/bin/bash
# Variables
seed=2
run_dir="ldm_v2_conditioned_v2" # Where your model is saved
training_ids="/data/dataset_train.tsv"
validation_ids="/data/dataset_validation.tsv"
vqvae_uri="[PATH TO MLRUNS FOLDER OF THE VAE]/artifacts/final_model" # Where the MLRUNS VAE model is saved.
# project points to the code (conditioned_ldm)
config_file="/project/configs/diffusion/ldm_v2_conditioned_v2.yaml"
lesions="wmh-tumour-edema"
batch_size=256
n_epochs=2000
eval_freq=200
augmentation=0
num_workers=32
experiment="NEWPREPROCESS"

bash -c "bash /project/src/bash/start_script.sh
python3 /project/src/python/training_and_testing/train_ldm_v2_conditioned_pv.py \
  seed=${seed} \
  run_dir=${run_dir} \
  training_ids=${training_ids} \
  validation_ids=${validation_ids} \
  vqvae_uri=${vqvae_uri} \
  config_file=${config_file} \
  lesions=${lesions} \
  batch_size=${batch_size} \
  n_epochs=${n_epochs} \
  eval_freq=${eval_freq} \
  augmentation=${augmentation} \
  num_workers=${num_workers} \
  experiment=${experiment}"
