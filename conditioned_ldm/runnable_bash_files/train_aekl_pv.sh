# Variables
seed=2
run_dir="aekl_pv_v0" # Where your model and logs are saved (folder)
training_ids="/data//dataset_train.tsv"
validation_ids="/data/dataset_validation.tsv"
config_file="/project/configs/stage1/ae_kl_newdataset_pv.yaml"
batch_size=256
n_epochs=1000
eval_freq=200
augmentation=1
num_workers=32
experiment="NEWPREPROCESS"

bash -c "bash /project/src/bash/start_script.sh
python3 /project/src/python/training_and_testing/train_ae_kl_pv_v1.py \
  seed=${seed} \
  run_dir=${run_dir} \
  training_ids=${training_ids} \
  validation_ids=${validation_ids} \
  config_file=${config_file} \
  batch_size=${batch_size} \
  n_epochs=${n_epochs} \
  eval_freq=${eval_freq} \
  augmentation=${augmentation} \
  num_workers=${num_workers} \
  experiment=${experiment}"
