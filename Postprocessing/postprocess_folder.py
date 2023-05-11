import numpy as np
import nibabel as nib
from os.path import join
from os import listdir
from segmentation_and_postprocessing.postprocessing import postprocess
from pathlib import Path

# Goal: take in a folder of output softmax.npz files and turn them into .nii.gz segmentation files


def save_segmentation_as_niigz(segmentation,ref_path,new_path):
    # Save .npz segmentation in nifti format using a reference image 

    ref_img = nib.load(ref_path)

    ref_hdr = ref_img.header
    ref_affine = ref_img.affine
    segmentation = np.swapaxes(segmentation,0,2)
    img1 = nib.Nifti1Image(segmentation, ref_affine,ref_hdr)
    print("Saving to:",new_path)
    nib.save(img1, new_path) 



def post_process_folder(npz_folder,output_folder_base,ref_im_folder,run_name,ROI_weight,seg_threshhold,post_process_ROIs,power):

    # Path formatting
    filenames = sorted(listdir(npz_folder))
    npz_filenames = [fname for fname in filenames if fname[-4:]==".npz"] # convenience
    npz_names = [join(npz_folder,fname) for fname in filenames if fname[-4:]==".npz"]
    ref_im_names = sorted(listdir(ref_im_folder))
    output_folder = join(output_folder_base,run_name)

    new_paths = []
    ref_im_paths = []

    #Makes output folder if it doesn't exist yet
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for i,npz_name in enumerate(npz_names):

        # Creates paths for referenfce and output image
        ref_im_path = join(ref_im_folder,ref_im_names[i])
        new_path = join(output_folder,npz_filenames[i][:-4]+".nii.gz")

        new_paths.append(new_path)
        ref_im_paths.append(ref_im_path)

        #Creates and saves segmentation
        segmentation, softmax, region_size = postprocess(npz_name,ROI_weight=ROI_weight,seg_threshhold=seg_threshhold,post_process_ROIs=post_process_ROIs,power=power)
        save_segmentation_as_niigz(segmentation,ref_im_path,new_path)

    return new_paths,ref_im_paths,npz_names




if __name__ == "__main__":

    run_name = "chamos_on_naxiva"
    ROI_weight = [0.95,1.10,1.10,0.95]
    seg_threshhold = 0.25
    power = 2

    run_name = run_name +"_"+ str(seg_threshhold) +"_"+ str(ROI_weight) +"_"+ str(power)

    separate_ROIs = False

    npz_folder = r"C:\Users\halja\Desktop\RESULTS\data\chamos\chamos_on_NAXIVA_NEW\nnUNetTrainerV2__nnUNetPlansv2.1"
    output_folder_base = r"C:\Users\halja\Desktop\RESULTS\data\chamos\postprocessed"
    ref_im_folder = r"C:\Users\halja\Desktop\RESULTS\data\images\cropped" #Do these need to be cropped images? I don' know...

    filenames = sorted(listdir(npz_folder))
    npz_filenames = [fname for fname in filenames if fname[-4:]==".npz"] # convenience
    npz_names = [join(npz_folder,fname) for fname in filenames if fname[-4:]==".npz"]
    ref_im_names = sorted(listdir(ref_im_folder))
    output_folder = join(output_folder_base,run_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for i,npz_name in enumerate(npz_names):
        segmentation, softmax, region_size = postprocess(npz_name,ROI_weight=ROI_weight,seg_threshhold=seg_threshhold,power=power)
        ref_im_path = join(ref_im_folder,ref_im_names[i])
        if separate_ROIs:
            for k in range(1,5):
                output = softmax[k,:,:,:] * 1000
                new_path = join(output_folder,(npz_filenames[i][:-9]+"_%04d"+".nii.gz") % k)
                save_segmentation_as_niigz(output,ref_im_path,new_path)
        else:
            new_path = join(output_folder,(npz_filenames[i][:-9]+"_0000"+".nii.gz"))
            print(new_path)
            save_segmentation_as_niigz(segmentation,ref_im_path,new_path) 
