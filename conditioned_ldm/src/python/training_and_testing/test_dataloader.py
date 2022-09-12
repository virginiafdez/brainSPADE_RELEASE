from util import get_training_data_loader_pv_conditioned_vol

path_to_train_data = "/data/dataset_train.tsv"
path_to_valid_data = "/data/dataset_validation.tsv"
num_workers = 12
batch_size = 8
lesions = ['wmh', 'tumour', 'edema']

print("Getting data...")

train_loader, val_loader = get_training_data_loader_pv_conditioned_vol(
    batch_size=batch_size,
    training_ids=path_to_train_data,
    validation_ids=path_to_valid_data,
    augmentation=True,
    num_workers=num_workers,
    lesions = lesions
)



