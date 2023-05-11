import nibabel as nib
import numpy as np
from os.path import join, basename, isfile
from os import listdir

### Crop niigz image to either nifti or .npz

# have to remember axis order is:

# (z,y,x) in the .npz
# (x,y,z) in the .nii.gz


def crop_niigz_to_niigz(nifti_path,output_folder):

    # Load .nii.gz
    data = nib.load(nifti_path)

    img = data.get_fdata()



    hdr = data.header
    affine = data.affine

    # Crop columns/rows that only contain 0 values (==sum to 0? since only +ve values?)
    
    #Axis 0
    rowsums = np.sum(img,axis=(1,2))
    init_sum = np.sum(rowsums)
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    img = img[crop_index_L:crop_index_R,:,:]

    #Axis 1
    rowsums = np.sum(img,axis=(0,2))
    init_sum = np.sum(rowsums)
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    img = img[:,crop_index_L:crop_index_R,:]

    #Axis 2
    rowsums = np.sum(img,axis=(0,1))
    init_sum = np.sum(rowsums)
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    img = img[:,:,crop_index_L:crop_index_R]


    final_sum = np.sum(img)

    print("Cropped shape:",img.shape)
    print(init_sum,final_sum)
    assert final_sum == init_sum, "These really should be the same!!!"
    
    # Save as .niigz in output_folder with the same filename

    image_name = basename(nifti_path)
    new_path = join(output_folder,image_name)


    img1 = nib.Nifti1Image(img, affine,hdr)
    nib.save(img1, new_path) 

def crop_niigz_to_npz(nifti_path,output_folder):

    # Load .nii.gz
    data = nib.load(nifti_path)

    img = data.get_fdata()

    # Crop columns/rows that only contain 0 values (==sum to 0? since only +ve values?)
    
    #Axis 0
    rowsums = np.sum(img,axis=(1,2))
    init_sum = np.sum(rowsums)
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    img = img[crop_index_L:crop_index_R,:,:]

    #Axis 1
    rowsums = np.sum(img,axis=(0,2))
    init_sum = np.sum(rowsums)
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    img = img[:,crop_index_L:crop_index_R,:]

    #Axis 2
    rowsums = np.sum(img,axis=(0,1))
    init_sum = np.sum(rowsums)
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    img = img[:,:,crop_index_L:crop_index_R]


    final_sum = np.sum(img)
    
    img = np.asarray(img).reshape((1,img.shape[0],img.shape[1],img.shape[2]))

    print("Cropped shape:",img.shape)
    print(init_sum,final_sum)
    assert final_sum == init_sum, "These really should be the same!!!"
    
    # Save as .npz in output_folder with the same filename

    image_name = basename(nifti_path).replace(".nii.gz",".npz")
    new_path = join(output_folder,image_name)

    np.savez(new_path,data=img)

def determine_empty_bounds(rowsums):

    # Determine, which rows / columns are the bound for empty data in the images
    dim = len(rowsums)
    print(dim)
    crop_index_L = 0
    crop_index_R = 0
    for i in range(dim):
        if rowsums[i] != 0:
            break
        else:
            crop_index_L += 1
    for i in range(dim-1,-1,-1):
        if rowsums[i] != 0:
            break
        else:
            crop_index_R -= 1

    print(crop_index_L,crop_index_R)

    if crop_index_R == 0:
        crop_index_R = dim
    return crop_index_L , crop_index_R

def crop_segmentation_using_image(seg_data,image_data):
    # Crop segmentation down to same size as it's corresponding image

    #Axis 0
    rowsums = np.sum(image_data,axis=(1,2))
    init_sum = np.sum(seg_data)
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    seg_data = seg_data[crop_index_L:crop_index_R,:,:]

    #Axis 1
    rowsums = np.sum(image_data,axis=(0,2))
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    seg_data = seg_data[:,crop_index_L:crop_index_R,:]

    #Axis 2
    rowsums = np.sum(image_data,axis=(0,1))
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    seg_data = seg_data[:,:,crop_index_L:crop_index_R]

    final_sum = np.sum(seg_data)

    print("Cropped shape:",seg_data.shape)
    print(init_sum,final_sum)
    assert final_sum == init_sum, "These really should be the same!!!"


    return seg_data

if __name__ == "__main__":
    
    seg_folder = r"C:\Users\halja\Desktop\MyProjectCode\NAXIVA\Organised dataset\new datadump for now\new labels"
    ref_im_folder = r"C:\Users\halja\Desktop\MyProjectCode\NAXIVA\Organised dataset\new datadump for now\new image niftis"
    out_folder_labels = r"C:\Users\halja\Desktop\MyProjectCode\NAXIVA\Organised dataset\new datadump for now\new labels cropped"
    out_folder_images = r"C:\Users\halja\Desktop\MyProjectCode\NAXIVA\Organised dataset\new datadump for now\new images cropped"
    

    seg_files = sorted([f for f in listdir(seg_folder) if isfile(join(seg_folder, f))])
    ref_im_files = sorted([f for f in listdir(ref_im_folder) if isfile(join(ref_im_folder, f))])

    for n,fname in enumerate(seg_files):
        seg = nib.load(join(seg_folder,fname))
        seg_data = seg.get_fdata()
        im_data = nib.load(join(ref_im_folder,ref_im_files[n])).get_fdata()

        seg_data_cropped = crop_segmentation_using_image(seg_data,im_data)
        print(seg_data_cropped.shape)

        hdr = seg.header
        affine = seg.affine
        img1 = nib.Nifti1Image(seg_data_cropped, affine,hdr)

        new_path = join(out_folder_labels,fname)
        nib.save(img1, new_path)

    for n,fname in enumerate(ref_im_files):
        crop_niigz_to_niigz(join(ref_im_folder,fname),out_folder_images)
