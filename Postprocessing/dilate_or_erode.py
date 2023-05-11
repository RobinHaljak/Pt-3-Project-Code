from scipy.ndimage import convolve
import numpy as np

# Dilate segmentation to include voxels within n layers (including connected by corner) around the boundary
def dilate(data,n):
    data = convolve(data,np.ones([2*n+1,2*n+1,2*n+1]),mode='constant',cval=0)
    data[data>0] = 1

    return data

# Erode segmentation to remove voxels within n layers (including connected by corner) around the boundary
def erode(data,n):
    
    data = convolve(data,np.ones([2*n+1,2*n+1,2*n+1]),mode='constant',cval=0)
    data[data!=(2*n+1)**3] = 0
    data[data==(2*n+1)**3] = 1

    return data