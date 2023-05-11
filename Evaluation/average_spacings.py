import numpy as np
from evaluation.physics import resolution
from os import listdir
from os.path import join
import nibabel as nib

# Determine average/median image voxel spacing (mm x mm x mm) and image size (voxels) for a folder.

image_folder = r"C:\Users\halja\Desktop\MyProjectCode\datasets\amos22\amos22_mri_validation_im"

# True if want to get results with coarse axis always along z
in_out_plane = False

filenames = [join(image_folder,x) for x in listdir(image_folder) if x[-7:] == ".nii.gz"]

x_spacings = []
y_spacings = []
z_spacings = []

x_sizes = []
y_sizes = []
z_sizes = []

for fname in filenames:
    print(fname)
    # Load data
    a = nib.load(fname)
    affine = a.affine
    data = a.get_fdata()

    # Save spacings & sizes
    x_sizes.append(data.shape[0])
    x_spacings.append(resolution(a)[0])

    # Align coarse axis with z
    if in_out_plane:
        if resolution(a)[1] < resolution(a)[2]:
            y_sizes.append(data.shape[2])
            z_sizes.append(data.shape[1])
        else:
            y_sizes.append(data.shape[1])
            z_sizes.append(data.shape[2])

        
        if resolution(a)[1] < resolution(a)[2]:
            y_spacings.append(resolution(a)[2])
            z_spacings.append(resolution(a)[1])

        else:
            y_spacings.append(resolution(a)[1])
            z_spacings.append(resolution(a)[2])
    else:
        y_sizes.append(data.shape[1])
        z_sizes.append(data.shape[2])
        y_spacings.append(resolution(a)[1])
        z_spacings.append(resolution(a)[2])

print(np.average(x_sizes),np.average(y_sizes),np.average(z_sizes))
print(np.average(x_spacings),np.average(y_spacings),np.average(z_spacings))

print(np.median(x_sizes),np.median(y_sizes),np.median(z_sizes))
print(np.median(x_spacings),np.median(y_spacings),np.median(z_spacings))
