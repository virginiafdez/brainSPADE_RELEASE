""" Script to preprocess the images from BRATS-TCIA using nipype.

In the preprocess, we perform the following step:
- 1) Register T1 to MNI space with rigid body + affine using ANTs RegistrationSynQuick
- 2) Use previous transformation to register FLAIR and masks to MNI space
"""
import argparse
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import HTTPError

from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.ants import RegistrationSynQuick


def main(args):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    template_dir = Path("/project/outputs/template/mni_icbm152_nlin_sym_09a_nifti/")

    # Download ICBM MNI152 Symmetric Template if it is missing
    if not template_dir.is_dir():
        temp_file = tempfile.gettempdir() + "/mni_icbm152_nlin_sym_09a_nifti.zip"
        template_url = "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip"
        try:
            urllib.request.urlretrieve(template_url, temp_file)
        except HTTPError as e:
            print('Error code: ', e.code)
        else:
            print("Template downloaded!")

        template_dir.mkdir(parents=True)

        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall(template_dir)
    else:
        print("Template already available!")

    template_dir = template_dir / "mni_icbm152_nlin_sym_09a"
    template_image = template_dir / "mni_icbm152_t1_tal_nlin_sym_09a.nii"
    template_mask = template_dir / "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
    space_name = "MNI152NLin2009aSym"

    sourcedata_dir = Path("/sourcedata/")
    derivatives_dir = Path("/targetdata/BRATS-TCIA_ANTs_RegistrationSynQuick/")
    derivatives_dir.mkdir(exist_ok=True)

    sourcedata_t1_dir = sourcedata_dir / "T1"
    sourcedata_t2_dir = sourcedata_dir / "T2"
    sourcedata_flair_dir = sourcedata_dir / "FLAIR"
    sourcedata_parcellation_dir = sourcedata_dir / "Parcellation"
    sourcedata_segmentation_dir = sourcedata_dir / "Segmentation"

    derivatives_t1_dir = derivatives_dir / "T1"
    derivatives_t2_dir = derivatives_dir / "T2"
    derivatives_flair_dir = derivatives_dir / "FLAIR"
    derivatives_parcellation_dir = derivatives_dir / "Parcellation"
    derivatives_segmentation_dir = derivatives_dir / "Segmentation"

    derivatives_t1_dir.mkdir(exist_ok=True)
    derivatives_t2_dir.mkdir(exist_ok=True)
    derivatives_flair_dir.mkdir(exist_ok=True)
    derivatives_parcellation_dir.mkdir(exist_ok=True)
    derivatives_segmentation_dir.mkdir(exist_ok=True)

    # Note: Preprocess the dataset in systems that use the same file sorting
    participant_list = sorted([f.stem[3:].split(".")[0] for f in sourcedata_t1_dir.glob("*.nii.gz")])

    for i, participant in enumerate(participant_list[args.start:args.stop]):
        participant_id = participant.split("_")[3]

        t1_image = sourcedata_t1_dir / f"T1_BraTS20_BRATS-TCIA_Training_{participant_id}.nii.gz"
        t2_image = sourcedata_t2_dir / f"T2_BraTS20_BRATS-TCIA_Training_{participant_id}.nii.gz"
        flair_image = sourcedata_flair_dir / f"FLAIR_BraTS20_BRATS-TCIA_Training_{participant_id}.nii.gz"
        parcellation_image = sourcedata_parcellation_dir / f"Parcellation_BraTS20_BRATS-TCIA_Training_{participant_id}.nii.gz"
        segmentation_image = sourcedata_segmentation_dir / f"Parcellation_BraTS20_BRATS-TCIA_Training_{participant_id}.nii.gz"

        try:
            # Register T1 to MNI space
            reg = RegistrationSynQuick()
            reg.inputs.fixed_image = str(template_image)
            reg.inputs.moving_image = str(t1_image)
            reg.inputs.transform_type = "a"
            reg.inputs.output_prefix = str(
                derivatives_dir / f"{participant_id}_space-{space_name}_T1w_"
            )
            reg.inputs.precision_type = "float"
            reg.inputs.num_threads = 2
            reg.inputs.args = f"-x {str(template_mask)}"
            print(reg.cmdline)
            reg.run()

            # Apply transformation to register T2 to MNI space
            at1 = ApplyTransforms()
            at1.inputs.dimension = 3
            at1.inputs.input_image = str(t2_image)
            at1.inputs.reference_image = str(template_image)
            at1.inputs.output_image = str(
                derivatives_t2_dir / f"T2_{participant_id}_space-{space_name}.nii.gz"
            )
            at1.inputs.transforms = [
                str(derivatives_dir / f"{participant_id}_space-{space_name}_T1w_0GenericAffine.mat")
            ]
            at1.inputs.float = True
            print(at1.cmdline)
            at1.run()

            # Apply transformation to register FLAIR to MNI space
            at1 = ApplyTransforms()
            at1.inputs.dimension = 3
            at1.inputs.input_image = str(flair_image)
            at1.inputs.reference_image = str(template_image)
            at1.inputs.output_image = str(
                derivatives_flair_dir / f"FLAIR_{participant_id}_space-{space_name}.nii.gz"
            )
            at1.inputs.transforms = [
                str(derivatives_dir / f"{participant_id}_space-{space_name}_T1w_0GenericAffine.mat")
            ]
            at1.inputs.float = True
            print(at1.cmdline)
            at1.run()

            # Apply transformation to register Parcellation to MNI space
            at1 = ApplyTransforms()
            at1.inputs.dimension = 3
            at1.inputs.input_image = str(parcellation_image)
            at1.inputs.reference_image = str(template_image)
            at1.inputs.output_image = str(
                derivatives_parcellation_dir / f"Parcellation_{participant_id}_space-{space_name}.nii.gz"
            )
            at1.inputs.transforms = [
                str(derivatives_dir / f"{participant_id}_space-{space_name}_T1w_0GenericAffine.mat")
            ]
            at1.inputs.args = f"-e 3"
            at1.inputs.float = True
            print(at1.cmdline)
            at1.run()

            # Apply transformation to register Segmentation to MNI space
            at1 = ApplyTransforms()
            at1.inputs.dimension = 3
            at1.inputs.input_image = str(segmentation_image)
            at1.inputs.reference_image = str(template_image)
            at1.inputs.output_image = str(
                derivatives_segmentation_dir / f"Parcellation_{participant_id}_space-{space_name}.nii.gz"
            )
            at1.inputs.transforms = [
                str(derivatives_dir / f"{participant_id}_space-{space_name}_T1w_0GenericAffine.mat")
            ]
            at1.inputs.args = f"-e 3"
            at1.inputs.float = True
            print(at1.cmdline)
            at1.run()
        except:
            print(f"{i} - {participant_id}: RegistrationSynQuick failed.")

        t1_reg_file = derivatives_t1_dir / f"T1_{participant_id}_space-{space_name}.nii.gz"
        (derivatives_dir / f"{participant_id}_space-{space_name}_T1w_Warped.nii.gz").rename(t1_reg_file)

        os.remove(str(derivatives_dir / f"{participant_id}_space-{space_name}_T1w_0GenericAffine.mat"))
        os.remove(str(derivatives_dir / f"{participant_id}_space-{space_name}_T1w_InverseWarped.nii.gz"))

        print(f"{i}: {participant_id} OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, help="Starting subject index to process.")
    parser.add_argument("--stop", type=int, help="Stopping subject index to process.")
    args = parser.parse_args()
    main(args)
