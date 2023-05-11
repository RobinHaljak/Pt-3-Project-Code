
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from scipy.ndimage.filters import gaussian_filter
import nibabel as nib

# A region expansion script for the VTT segmentation. Not used in the final project.



def dice_score(seg1, seg2):
    # Determine dice score
    intersection = np.sum(seg1 * seg2)
    union = np.sum(seg1) + np.sum(seg2)
    dice = (2. * intersection) / (union + 1e-9) # adding a small epsilon to avoid division by zero
    return dice

def region_grow(image, mask, gt_seg, intensity_threshold, gradient_threshold,stop_growth=10,num=25):

    # Grow region to voxel if it is within certain intensity range and is not on a boundary
    # determined by the gradient.

    # Define the neighborhood connectivity and gradient operator
    neighborhood = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                             [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    gradient = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                     [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                     [[1, 0, -1], [2, 0, -2], [1, 0, -1]]])

    # Iterate until the mask stops growing
    mask_old = np.zeros_like(mask)
    mask_init = np.copy(mask)
    n = 1
    l = 4 # Steps between weak gaussian blur
    m = 15 # Steps between strong gaussian blur 
    dice_scores = []
    ROI_sizes = [mask_init.sum()]
    while not np.array_equal(mask_old, mask): # Stop if mask stops growing iteratively
        print(" ")
        print(n)

        mask_old = np.copy(mask)

        # Identify the border voxels
        border_voxels = np.logical_and(np.logical_not(mask), ndimage.convolve(mask.astype(np.int16), neighborhood, mode='constant') > 0)


        border_gradient = np.abs(ndimage.convolve(image, gradient, mode='nearest'))[border_voxels]

        # Calculate the mean intensity within the mask
        mean_intensity = np.mean(image[mask])
        print(mean_intensity)

        # Add voxels to the mask that meet the intensity and gradient criteria

        print(border_voxels.sum())
        mask[border_voxels] = np.logical_and(
            np.logical_and(
                np.abs(image[border_voxels] - mean_intensity) < intensity_threshold,
                border_gradient < gradient_threshold
            ),
            np.logical_not(mask[border_voxels])
        )

        # Check  for termination conditions
        if(mask.sum() < mask_old.sum() + stop_growth):
            break
        if n > num:
            break
        
        
        # Gaussian blur
        if n % m == m-1:
            mask = gaussian_filter(mask.astype(float), sigma=2) > 0.4

        elif n % l == l-1:
            mask = gaussian_filter(mask.astype(float), sigma=1) > 0.5

        mask = (mask+mask_init) >= 1


        n+=1
        print(mask_old.sum(),mask.sum())
        
        dice = dice_score(gt_seg,mask)
        print(dice)
        dice_scores.append(dice)
        ROI_sizes.append(mask.sum())
    return mask, dice_scores, ROI_sizes


# Image & segmentation paths
image_file = r"C:\Users\halja\Desktop\RESULTS\data\images\cropped\NAXIVA_0017_0000.nii.gz"
seg_file = r"C:\Users\halja\Desktop\MyProjectCode\ES_25_ensemble_output_weight2\NAXIVA_0017_0000.nii.gz"
GT_seg_file = r"C:\Users\halja\Desktop\RESULTS\data\labels\cropped\NAXIVA_0017_0000.nii.gz"

# Load data
img = nib.load(image_file)
seg = nib.load(seg_file)
gt = nib.load(GT_seg_file)

img_data = img.get_fdata()
seg_data = seg.get_fdata().astype(bool)
gt_data = gt.get_fdata().astype(bool)

assert img_data.shape == seg_data.shape == gt_data.shape, "All images should be cropped"
print("Shape:",img_data.shape)


# Find stddev of ROI voxels

flat_ROI = (seg_data*img_data).flatten()
flat_ROI = flat_ROI[np.where(flat_ROI!=0)]

stddev = np.std(flat_ROI)
print(stddev)

# Define the threshold parameters
intensity_threshold = stddev*2
gradient_threshold = 1000
stop_growth = 20

# Apply the region growing algorithm
output_mask, dice_scores, ROI_sizes = region_grow(img_data, seg_data, gt_data, intensity_threshold, gradient_threshold,stop_growth=stop_growth,num=50)

# apply Gaussian smoothing to mask
sigma = 2 # adjust sigma as needed
filter_threshhold = 0.5
smoothed_mask = gaussian_filter(output_mask.astype(float), sigma=sigma) > filter_threshhold


img_data = np.flip(np.swapaxes(img_data,0,1),1)
seg_data = np.flip(np.swapaxes(seg_data,0,1),1)
output_mask = np.flip(np.swapaxes(output_mask,0,1),1)
smoothed_mask = np.flip(np.swapaxes(smoothed_mask,0,1),1)

fig, axes = plt.subplots(1, 2)
axes[0].plot(dice_scores)
axes[0].set_title('Dice scores')
axes[1].plot(ROI_sizes)
axes[1].set_title('ROI_sizes')
plt.show()



# Plot the input image and the resulting mask
fig, axes = plt.subplots(1, 3)
axes[0].imshow(img_data[:,:,47], cmap='gray',origin='lower')
axes[0].set_title('Input Image')
axes[1].imshow(output_mask[:,:,47],origin='lower')
axes[1].set_title('Output Mask')
axes[2].imshow(seg_data[:,:,47],origin='lower')
axes[2].set_title('Output Mask')
plt.show()

# Plot the input image and the resulting mask
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_data[:,:,47], cmap='gray',origin='lower')
axes[0].set_title('Input Image')
axes[1].imshow(smoothed_mask[:,:,47],origin='lower')
axes[1].set_title('Smoothed Mask')
plt.show()

combined_mask = output_mask + seg_data

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_data[:,:,47], cmap='gray',origin='lower')
axes[0].set_title('Input Image')
axes[1].imshow(combined_mask[:,:,47],origin='lower')
axes[1].set_title('Smoothed Mask')
plt.show()