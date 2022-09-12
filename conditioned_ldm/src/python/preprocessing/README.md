# Preprocessing
In the preprocessing, we register the images (originally from the prepoc folders), to the MNI space using rigid body + affine transformations.
The preprocessing is performed using ANTS (version 2.3.4) using the function RegistrationSynQuick (https://github.com/ANTsX/ANTs/blob/master/Scripts/antsRegistrationSyNQuick.sh).
For the template, we are using ICBM 2009a Nonlinear Symmetric (from https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009 from mni_icbm152_nlin_sym_09a_nifti.zip file).

Each dataset has a different script since they had different data available.
 