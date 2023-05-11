import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import label



# Histogram of softmax values
def softmax_histogram(softmax,ROI_index,logged = True):
    image = softmax[ROI_index,:,:,:]
    values = image.flatten()
    #values = np.delete(values, np.where(values == 0), axis=0)
    num_softmax_voxels = values.shape[0]

    bin_bounds,bins,x= plt.hist(values,bins=100,log=logged)
    plt.show()
    return bin_bounds, bins

# Split segmentation into left / right halves in image
def Left_Right_softmax_split(softmax):
    # assumes R kidney = 2, L kidney = 3
    L_plus_R = softmax[2,:, :, :] + softmax[3,:, :, :] 
    img_shape = softmax.shape
    x_mid = int(softmax.shape[3]/2)

    L_softmax = np.concatenate((L_plus_R[:,:,:x_mid],np.zeros([img_shape[1],img_shape[2],img_shape[3]-x_mid])),axis=2)
    R_softmax = np.concatenate((np.zeros([img_shape[1],img_shape[2],x_mid]),L_plus_R[:,:,x_mid:]),axis=2)
    
    # For now say middle of image is the middle voxel -- NOT NECESSARILY TRUE!!!! Set X_mid using affine transform
    softmax[2,:,:,:] = R_softmax
    softmax[3,:,:,:] = L_softmax

    assert (softmax[2,:,:,:] + softmax[3,:,:,:] == L_plus_R).all(), "expect these to be equal"

    return softmax



def keep_largest_region(softmax,lmap,region_probabilities,ROI):
    # Keep only the largest region in segmentation softmax
    max_region_size = max(region_probabilities)

    keep_index = region_probabilities.index(max_region_size)
    print("Keeping region:",keep_index)

    lmap_keep_region = lmap == keep_index + 1 #+1 since ignored background

    softmax[ROI,:,:,:] = np.multiply(softmax[ROI,:,:,:],lmap_keep_region)

    return softmax

def keep_largest_regions_thresholded(softmax,lmap,region_probabilities,ROI,softmax_threshhold_ratio=0.5,verbose=True):
    # Keep all regions with cumulative probabilities over softmax_threshold of maximum probability
    max_region_size = max(region_probabilities)
    threshhold_region_size = softmax_threshhold_ratio*max_region_size

    keep_indices = []
    for n, size in enumerate(region_probabilities):
        if size > threshhold_region_size:
            keep_indices.append(n)

    if verbose:
        print("Keeping region(s):",keep_indices)

    lmap_keep_region = np.zeros(softmax[0,:,:,:].shape)
    for keep_index in keep_indices:
        lmap_keep_region += lmap == keep_index + 1 #+1 since ignored background

    softmax[ROI,:,:,:] = np.multiply(softmax[ROI,:,:,:],lmap_keep_region)

    return softmax


def determine_regions(softmax,seg_threshhold,ROI,verbose=True):
    #Determine regions by connected components analysis of spftmax
    segmentation_init = softmax[ROI,:, :, :] > seg_threshhold
    
    lmap, num_objects = label(segmentation_init.astype(int))
    if verbose:
        print("ROI:",ROI,"Voxels segmented:",segmentation_init.sum())
        print("Number of objects:",num_objects)
    unique, counts = np.unique(lmap, return_counts=True)
    return lmap, unique, counts

def determine_region_probabilities(softmax,lmap,unique,counts,ROI,power=3):
    # Determine cumulative probabilities of each region using softmax probabilities
    region_probabilities = []
    avg_probabilities = []
    for region in unique:
        if region != 0:
            lmap_region = lmap == region
            softmax_region = np.multiply(lmap_region,softmax[ROI,:,:,:])**power
            region_softmax = softmax_region.sum()
            #print("Region id:",region,"softmax:",region_softmax)

            region_probabilities.append(region_softmax)
            avg_probabilities.append(round(region_softmax/counts[region],4)) # Could replace region
    if np.array(region_probabilities).argmax(0) != np.array(avg_probabilities).argmax(0):
        print(" ")
        print("WARNING -- LARGEST SOFTMAX REGION IS NOT THE HIGHEST AVERAGE SOFTMAX")
        print("Consider using a lenient softmax region selection")
        print(" ")
    return region_probabilities, avg_probabilities

def segment(softmax,ROI_weight=1,verbose=True):
    # ROI_weight is the artificial (over)confidence factor of the segmentation, 1 is standard confidence
    if type(ROI_weight) == list:
        for k in range(1,softmax.shape[0]):
            softmax[k:,:,:,:] *= ROI_weight[k-1]
    else:
        softmax[1:,:,:,:] *= ROI_weight

    segmentation = softmax.argmax(0) # Give class of highest confidence in post-processed softmax
    
    regions, counts = np.unique(segmentation,return_counts=True)
    
    region_sizes = dict(zip(regions, counts))
    
    if verbose:
        print("Created segmentation!")
        print("ROIs & Counts:")
        print(region_sizes)
    return segmentation,region_sizes





def postprocess(npz_path,seg_threshhold=0.01,ROI_weight=1.0,post_process_ROIs =[1,2,3,4],left_right_pooling=True,keep_only_largest_region=True,threshholded_largest_only=False,threshholded_largest_only_ratio=0.5,power=2,verbose=True):

    # Post-process image:
    # 1. Left-right pooling (kidneys)
    # 2. Keep most probable region(s)
    # 3. Create segmentation

    data = np.load(npz_path)
    softmax = data["softmax"].astype(np.float32)
    
    if verbose:
        print("Image shape:",softmax[0,:, :, :].shape)
    if left_right_pooling and 2 in post_process_ROIs and 3 in post_process_ROIs:
        softmax = Left_Right_softmax_split(softmax) # For future could rewrite so can give any two indices to L/R pool, for now no point


    for i in post_process_ROIs:
        #softmax_histogram(softmax,i)
        lmap, unique, counts = determine_regions(softmax,seg_threshhold,i,verbose=verbose)
        region_probabilities, avg_probabilities = determine_region_probabilities(softmax,lmap,unique,counts,i,power=power)
        if keep_only_largest_region:
            if threshholded_largest_only:
                softmax = keep_largest_regions_thresholded(softmax,lmap,region_probabilities,i,softmax_threshhold_ratio=threshholded_largest_only_ratio,verbose=verbose)
            else:
                softmax = keep_largest_region(softmax,lmap,region_probabilities,i) ## Can replace with any logic for keeping connected regions (e.g. size threshhold)
    segmentation, region_sizes = segment(softmax,ROI_weight,verbose=verbose)   

    return segmentation, softmax, region_sizes
    



if __name__ == '__main__':
    npz_file = "" # .npz of the softmax to postproccess
    gt_seg_file = "" #neccessary for dice score calculation
    im_file = "" #.nii.gz -- necessary only for the affine transformation 
    
    postprocess(npz_file,seg_threshhold=0.01,left_right_pooling=True,ROI_weight=1.0,keep_only_largest_region=True
                ,threshholded_largest_only=True,threshholded_largest_only_ratio=0.5)










