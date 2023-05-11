import numpy as np
from segmentation_and_postprocessing.postprocessing import postprocess
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Determine volume uncertainty by assuming all voxels segmented randomly with a probability given by their softmax output value.

# NB! This apporach is NOT representative of the actual scale of uncertainty (see report)

def Monte_Carlo_volume(softmax,ROI_index,num_simulations):
    # Calculate volume uncertainty of a ROI by monte carlo process
    image = softmax[ROI_index,:,:,:]
    values = image.flatten()
    values = np.delete(values, np.where(values == 0), axis=0)
    num_softmax_voxels = values.shape[0]

    MC_volumes = []
    for n in range(num_simulations):
        rand_array = np.random.rand(num_softmax_voxels) # Could setn all of them to have the same value for a more accurate representation of the results.

        larger_than = rand_array < values

        monte_carlo_volume = larger_than.sum()

        MC_volumes.append(monte_carlo_volume)

    return MC_volumes

# Take in post-processed softmax, already know the voxel volume of segmentation in that case, but can also calculate again for any ROI_weight if I wish
# Do monte carlo simulation

# 1. Extract all values larger than 0 (flatten, remove 0s): 
# 2. Anyway then generate random 0<x<1 and check how many are larger (or generate a different random for each one??)
# 3. Iterate some number of times, record volume results
# 4. Histogram p vs volume
# 5. fit to normal distribution, determine standard deviation
# 6. evaluate how normal the distribution is (Shapiro-Wilk)

if __name__ == "__main__":
    npz_name = "C:\\Users\\halja\\Desktop\\MyProjectCode\\507_on_NAXIVA\\nnUNetTrainerV2__nnUNetPlansv2.1\\NAXIVA_0000_0000.npz"
    ROI_weight = 1.00
    seg_threshhold = 0.03
    segmentation, softmax, region_size = postprocess(npz_name,ROI_weight=ROI_weight,seg_threshhold=seg_threshhold)
    ROI_index = 4
    num_simulations = 500
    MC_volumes = Monte_Carlo_volume(softmax/ROI_weight,ROI_index,num_simulations)

    avg_volume = np.average(MC_volumes)
    stdev_volume = np.std(MC_volumes)
    print("Average MC Volume:",round(avg_volume),"Standard deviation:",round(stdev_volume))
    print("Relative error:",str(round((100*stdev_volume/avg_volume),2))+"%")
    values,bins,x= plt.hist(MC_volumes,bins=25)
    
    # Test for normality
    print("Shapiro-Wilk p-value:",shapiro(MC_volumes))

    plt.show()
