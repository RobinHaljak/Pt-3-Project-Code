from math import erf
import numpy as np
from segmentation_and_postprocessing.postprocessing import postprocess
from scipy.stats import norm
from os import listdir
from os.path import join
import nibabel as nib

# Determine volumes and their uncertainties for segmentations

def std_dev_volume(filepath,std_dev,seg_threshold=0.00001):

    p_up = norm.cdf(std_dev, 0, 1)
    p_down = norm.cdf(-std_dev, 0, 1)

    ROI_up = p_up / (1-p_up)
    ROI_down = p_down / (1-p_down)


    segmentation_up, softmax , region_sizes = postprocess(filepath,seg_threshhold = seg_threshold,ROI_weight=ROI_up,post_process_ROIs=[1],verbose=False)
    segmentation, softmax, region_sizes = postprocess(filepath,seg_threshhold = seg_threshold,ROI_weight=1,post_process_ROIs=[1],verbose=False)
    segmentation_down, softmax, region_sizes = postprocess(filepath,seg_threshhold = seg_threshold,ROI_weight=ROI_down,post_process_ROIs=[1],verbose=False)

    vol_up = np.sum(segmentation_up)
    vol = np.sum(segmentation)
    vol_down = np.sum(segmentation_down)
    
    print("Standard deviations:",std_dev)
    print("Volume range:",vol_up,vol,vol_down)
    return vol_up,vol,vol_down

# Image input folder
npz_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results\510\val_RE_npz"
std_dev = 1
filenames = [join(npz_folder,x) for x in listdir(npz_folder) if x[-4:] == ".npz"]

for fname in filenames:
    print(fname[-20:])
    std_dev_volume(fname,std_dev)



