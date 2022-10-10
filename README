# brainSPADE
Reference:
Author: Virginia Fernandez (@virginiafdez) & Walter Hugo Lopez Pinaya (@warvito)
Institution: King's College London
Year: 2022

This repository contains 2 main sub-modules:
- conditioned_ldm: contains the code to train an LDM (based ona Spatial VAE + Diffusion model), with or without conditioning.
- image_generator: modified version of SPADE (https://github.com/NVlabs/SPADE.git)

Trainable files:
- image_generator: train.py; options are passed via commands (see options > base_options.py and options > train_options.py),
modisc in modality_classifier, to train the modality discriminator necessary if you want to use the modisc option in
brainSPADE. Once you train the modality classifier, you can pass the path to the network as argument mod_disc_path,
or the directory where it is as mod_disc_dir and the last epoch as mod_disc_epoch (i.e. if the file is called 40_net_MD.pth,
pass 40).

- conditioned_ldm: in src > python > training_and_testing > you will find train train_ae_kl_pv_v1.py (to train the Spatial
VAE - options in yaml format in configs > stage1 > ae_kl_newdataset_pv.yaml and in the script itself) and
train_ldm_v1.py and train_ldm_v2_conditioned_pv (to train the unconditioned and conditioned LDMs - options in yaml format
in configs > diffusion > ae_kl_nexp_pv.yaml and ae_kl_newdataset_pv.yaml and in the script itself).


## DATA FORMAT:

The data format is summarised in the attached images.

![plot](/home/vf19/Documents/JOURNAL2022/data_format_label_ge.png)

The datasets for the label generator are created as:
> train, validation and test folders, and in each of them: labels or pv_maps folders. In them, files are NPY,
and named [ROOT FILE: can be T1, FLAIR or Parcellation]_XXXX_XXXX_13_tumour-edema.npy. The "13" is the slice number
if the datasets are sliced; the last in between hyphen token is which diseases are present in the slice. In the filename,
the dataset name is also included (i.e. ADNI).
> labels and partial volume maps: CxHxW or CxHxWxD in case you want to implement a 3D version. Further description
of the format below.

The datasets for SPADE are:
> images_test, labels_test, images_train, labels_train and labels_validation and images_validation. All modalities are
together in the "images" folder. The naming of files is the same as previous, but the format is NPZ, as the files
also contain metadata (i.e. affine transforms, headers etc). You need to make sure that the dataset name is in the
list of datasets passed as argument "datasets" to SPADE when training.
> Images: they need to be of dimension HxWx1 (slices) and HxWxD (volumes). The npz has to have keys: 'img' (array with the image),
'img_affine' (affine transform from the nii file, can be an eye of 3x3), 'img_header': optional header of original nii file,
'dataset_name': i.e. ADNI, SABRE (string), 'slice': slice number if it's a sliced dataset, 'modality' (T1, FLAIR etc.)
> Labels: the specific format is described below. The NPZ have to have fields 'label', of dimension HxWx1xC (sliced) or
HxWxDxC (volume) where channel is the number of different regions, 'label_affine', affine transform of the nii file,
'label_header', optional header of the nii file, 'dataset_name', string with the dataset name, same as above,
'slice': slice number if the dataset type is sliced.

The partial volume maps or segmentation maps used for SPADE and the label generator must:
- Be one-hot-encoded: one tissue type per channel
- Be bound between 0 and 1
- Sum to 1 along the channel dimension

The images:
- Will be normalised in SPADE
- Ensure there aren't nans or inf values
- Make them float32 if possible

To train SPADE, the folder names are passed directly.
For the label generator, you need to pass a series of TSV files (one per section: train, validation and test).
These TSV files are created using the sh and py files in src > python > data_utils.

The partial volume maps and binary labels used for this project were obtained with GIF: Cardoso MJ, Modat M, Wolz R, Melbourne A, Cash D, Rueckert D, Ourselin S. Geodesic Information Flows: Spatially-Variant Graphs and Their Application to Segmentation and Fusion. IEEE Trans Med Imaging. 2015 Sep;34(9):1976-88. doi: 10.1109/TMI.2015.2418298. Epub 2015 Apr 14. PMID: 25879909.
