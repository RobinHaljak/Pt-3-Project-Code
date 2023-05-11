from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import nibabel as nib




def png2nifti_conversion(input_path,output_path,reference_path):

    # input_path is path of folder with all the .png images
    # output_path is path/filename.nii.gz of given segmentation
    # reference_path is path/ref_filename.nii.gz of corresponding volume

    ### Made for CHAOS case but can adapt/use for inspiration for other cases as well I guess
    files = listdir(input_path)
    num_files = len(files)

    # Assumes images are in top down order as they are in CHAOS case

    images = []


    for i in range(num_files):

        img = Image.open(join(input_path,files[i]))
        img_shape = np.asarray(img).shape
        images.append(img)
        if i == 0:
            shape = img_shape
        elif shape != shape:
            print("Oopsie, something has gone very very wrong (images in the input have to be of the same dimension)")
        
    pixData = np.zeros([shape[0],shape[1],num_files])

    #Take pixel values into 3D np voxel array
    for j in range(num_files):
        pixData[:,:,j] = np.asarray(images[j])

    ### For the CHAOS case need to apply this transform, don't ask me why..
    pixData = np.transpose(np.rot90((np.rot90(pixData))),(1,0,2))

    ### Converting pixel ranges into segmentation values (with amos organ indices):

    pixData[(pixData>=55) & (pixData<=70)] = 6 #Liver
    pixData[(pixData>=110) & (pixData<=135)] = 2 #R Kidney
    pixData[(pixData>=175) & (pixData<=200)] = 3 #L Kidney
    pixData[(pixData>=240) & (pixData<=255)] = 1 #Spleen


    ### Use corresponding volume file (converted to nifti from DICOM using dicom_to_nifti_my_ver.py)
    ### to use same affine / header

    ref_im = nib.load(reference_path)

    hdr = ref_im.header
    affine = ref_im.affine

    seg = nib.Nifti1Image(pixData,affine,hdr)

    nib.save(seg,output_path)


base_folder = "C:\\Users\\halja\\Desktop\\MyProjectCode\Chaos\\Train_Sets\\MR\\"

output_base = "C:\\Users\\halja\\Desktop\\MyProjectCode\\CHAOS_Converted\\"

im_indices = listdir(base_folder)

for im in im_indices:
    #1. T1DUAL (creates separate folders for validation for InPhase/OutPhase even though the segmentation is the same)

    print("T1DUAL, image: ", im)

    out_name_IN = "CHAOS_T1DUAL_IN_"+im+".nii.gz"
    out_name_OUT = "CHAOS_T1DUAL_OUT_"+im+".nii.gz"
    input_T1 = "C:\\Users\\halja\\Desktop\\MyProjectCode\\Chaos\\Train_Sets\\MR\\"+im+"\\T1DUAL\\Ground"
    output_T1_IN = "C:\\Users\\halja\\Desktop\\MyProjectCode\\CHAOS_Converted\\Segmentations\\T1DUAL_IN\\"+out_name_IN
    output_T1_OUT = "C:\\Users\\halja\\Desktop\\MyProjectCode\\CHAOS_Converted\\Segmentations\\T1DUAL_OUT\\"+out_name_OUT
    ref_T1 = "C:\\Users\\halja\\Desktop\\MyProjectCode\\CHAOS_Converted\\T1DUAL_IN\\CHAOS_T1DUAL_IN_"+im+".nii.gz"

    png2nifti_conversion(input_T1,output_T1_IN,ref_T1)
    png2nifti_conversion(input_T1,output_T1_OUT,ref_T1)

    #2. T2SPIR

    print("T2SPIR, image: ", im)
    
    out_name = "CHAOS_T2SPIR_"+im+".nii.gz"
    input_T2 = "C:\\Users\\halja\\Desktop\\MyProjectCode\\Chaos\\Train_Sets\\MR\\"+im+"\\T2SPIR\\Ground"
    output_T2 = "C:\\Users\\halja\\Desktop\\MyProjectCode\\CHAOS_Converted\\Segmentations\\T2SPIR\\"+out_name
    ref_T2 = "C:\\Users\\halja\\Desktop\\MyProjectCode\\CHAOS_Converted\\T2SPIR\\CHAOS_T2SPIR_"+im+".nii.gz"

    png2nifti_conversion(input_T2,output_T2,ref_T2)