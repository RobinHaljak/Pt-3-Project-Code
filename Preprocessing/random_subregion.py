
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from scipy.ndimage import gaussian_filter
import nibabel as nib
from preprocessing.precropping import save_cropped
from evaluation.physics import bbox2_3D, resolution
from os.path import join
from os import listdir

def get_random_voxels(mask, num_voxels=1):
    # Find all non-zero voxel coordinates in the mask
    nonzero_coords = np.transpose(np.nonzero(mask))
    # Randomly select num_voxels voxels from the non-zero coordinates
    random_coords = nonzero_coords[np.random.choice(nonzero_coords.shape[0], size=num_voxels, replace=True)]

    # Return the [x, y, z] coordinates of the selected voxels
    return random_coords.tolist()


def grow_subregion(mask,N,size_frac,resolution,gaussian_sigma=1.0,gaussian_sigma_final=2):

    #sigma = (gaussian_sigma/resolution[0],gaussian_sigma/resolution[1],0)
    #sigma_final = (gaussian_sigma_final/resolution[0],gaussian_sigma_final/resolution[1],0)

    sigma = (gaussian_sigma/resolution[0],gaussian_sigma/resolution[1],gaussian_sigma/resolution[2])
    sigma_final = (gaussian_sigma_final/resolution[0],gaussian_sigma_final/resolution[1],gaussian_sigma_final/resolution[2])
    #print(sigma,sigma_final)
    # Define the neighborhood connectivity operator
    neighborhood = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                             [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    
    # find initial voxel
    # voxel_init (x,y,z) coordinates of initial voxel (within ROI) to start growing from
    voxel_init = get_random_voxels(mask,num_voxels=1)[0]
    
    print("Initialising from voxel:",voxel_init)
    
    mask_old = np.zeros_like(mask)
    mask_init = np.copy(mask_old)
    mask_init[voxel_init[0],voxel_init[1],voxel_init[2]] = 1

    mask_size = np.sum(mask)
    

    n = 1 # Iteration counter
    m = 20 # Gaussian blurring counter
    k = 5 # one in k neighbouring voxels get added to the 


    for i in range(N):
        border_voxels = np.logical_and(np.logical_not(mask_init), ndimage.convolve(mask_init.astype(np.int16), neighborhood, mode='constant') > 0)

        
        #only want to choose from voxels within the mask
        border_voxels = border_voxels * mask

        N_border = int(np.ceil(np.sum(border_voxels) / k))
        

        voxels_ad = get_random_voxels(border_voxels,num_voxels=N_border)

        for voxel in voxels_ad:
            mask_init[voxel[0],voxel[1],voxel[2]] = 1

        if i % m == m-1:
            mask_init = gaussian_filter(mask_init.astype(float), sigma=sigma) > 0.5
            #print("Gaussian blur")
        #mask_init = mask * mask_init

        submask_size = np.sum(mask_init)
        #print("Mask size:",submask_size)
        if submask_size >= mask_size*size_frac:
            mask_init = gaussian_filter(mask_init.astype(float), sigma=sigma_final)
            submask_size = np.sum(mask_init)
            print("Mask size:",int(submask_size))
            print(i,"iterations")
            print("Desired/achieved size fraction",submask_size/mask_size,size_frac)
            return mask_init

        
    mask_init = gaussian_filter(mask_init.astype(float), sigma=sigma_final)
    submask_size = np.sum(mask_init)
    print("Mask size:",int(submask_size))
    print(i,"iterations")
    print("Desired/achieved size fraction",submask_size/mask_size,size_frac)
    return mask_init

def create_subbatches_image(image_file,GT_seg_file,out_folder_im,out_folder_seg,out_folder_subvol,index):

    img = nib.load(image_file)
    gt = nib.load(GT_seg_file)

    img_data = img.get_fdata()
    gt_data = gt.get_fdata().astype(bool)
    img_shape = img_data.shape
    img_affine = img.affine
    img_header = img.header

    img_res = resolution(img)

    ROI_vol = np.sum(gt_data)
    num_subvols = int(ROI_vol**(1/3))
    fillings = np.array([x for x in range(1,num_subvols+1)])/(num_subvols+2)
    print("Image:",image_file)
    print(num_subvols,"Subvolumes")
    
    assert img_data.shape == gt_data.shape, "All images should be cropped"
    print("Shape:",img_data.shape)
    print("Image resolution:",img_res)
    print("GT mask size:",ROI_vol)

    
    for j in range(num_subvols):
        out_im = join(out_folder_im,"REGION_"+("%04d" % int(index))+".nii.gz")
        out_seg = join(out_folder_seg,"REGION_"+("%04d" % int(index))+"_0000.nii.gz")
        out_subvol = join(out_folder_subvol,"REGION_"+("%04d" % int(index))+"_0001.nii.gz")

        output_mask = grow_subregion(gt_data,500,fillings[j],img_res,gaussian_sigma=0.6,gaussian_sigma_final=0.7)

        output_mask_borders = bbox2_3D(output_mask)

        ratio = 19
        buffer_ = (int(output_mask_borders[0]-img_shape[0]//ratio),
                int(output_mask_borders[1]+img_shape[0]//ratio),
                int(output_mask_borders[2]-img_shape[1]//ratio),
                int(output_mask_borders[3]+img_shape[1]//ratio),
                int(output_mask_borders[4]-img_shape[2]//ratio),
                int(output_mask_borders[5]+img_shape[2]//ratio))

        buffer = {}
        buffer[0] = buffer_ 
        #Saving results
        save_cropped(img_data,output_mask,buffer,output_mask,out_im,out_subvol,img_affine,img_header,0)
        save_cropped(img_data,gt_data,buffer,gt_data,out_im,out_seg,img_affine,img_header,0)

        index += 1
    

    return index


if __name__ == "__main__":

    im_folder = r"C:\Users\halja\Desktop\RESULTS\data\precropping_output\precropping_chamos_on_naxiva_0.2_[1.0, 1.0, 1.0, 1.0]_2_1\images"
    seg_folder = r"C:\Users\halja\Desktop\RESULTS\data\precropping_output\precropping_chamos_on_naxiva_0.2_[1.0, 1.0, 1.0, 1.0]_2_1\labels"

    im_out_folder = r"C:\Users\halja\Desktop\RESULTS\data\region_expansion_training\images"
    seg_out_folder = r"C:\Users\halja\Desktop\RESULTS\data\region_expansion_training\segmentations"
    subvol_out_folder = r"C:\Users\halja\Desktop\RESULTS\data\region_expansion_training\subvolumes"

    training_IDs = [42,0,1,2,3,4,5,11,12,24,25,26,33,34,13,14,46,20,21,22,23,18,19,30,31,32,27,28,35,36,37,38,6,7,45]

    index = 0
    indexes = []
    image_files = sorted(listdir(im_folder))

    for n,imfile in enumerate(image_files):
        if n in training_IDs: # change to not in to create test set instead
            index_old = index
            img_path = join(im_folder,imfile)
            seg_path = join(seg_folder,imfile)
            index = create_subbatches_image(img_path,seg_path,im_out_folder,seg_out_folder,subvol_out_folder,index)
            indexes.append([index_old,index])
            
    print(indexes)