import os
import numpy as np
import nibabel as nib

# Calculate dice score between two segmentations.

def dice_score(file_gt,file_pr,verbose=True,indices_pairs = [[None,None]]):
    # indices_pairs allows to give the indices of the ROIs (the values in the nifti segmentation files) between which the dice score is calculated.

    # Allows for inputting an already loaded numpy voxel image instead of loading from file.
    if type(file_gt) != np.ndarray:
        img_gt = nib.load(file_gt)
        data_gt = img_gt.get_fdata()
    else:
        data_gt = file_gt
    if type(file_pr) != np.ndarray:
        img_pr = nib.load(file_pr)
        data_pr = img_pr.get_fdata()
    else:
        data_pr = file_pr

    if data_gt.shape == data_pr.shape:
        num_labels = int(np.amax(data_gt))
        if (indices_pairs[0][0]==None and indices_pairs[0][1]==None): #Exclude background (by default assume label 0)
            dice_scores = np.zeros([num_labels])
            for i in range(1,num_labels+1):

                #Determine True positives
                data_gt_i = data_gt == i
                data_pr_i = data_pr == i
                data_tp_i = (data_gt_i == data_pr_i)*data_gt_i

                S1 = np.sum(data_gt_i)
                S2 = np.sum(data_pr_i)
                TP = np.sum(data_tp_i) 
                FP = S2 - TP
                sens = TP/S1 #sensitivity
                precision = np.nan_to_num(TP/S2) # precision

                dice_scores[i-1]=calc_dice(TP,S1,S2)
                if(verbose):
                    print(f"Dice score for index {i} is {round(dice_scores[i-1],3)}")
                    print("Gound Truth:",S1,"Segmentation:",S2,"True Positive:",TP)
                    print("Sensitivity:",round(TP/S1,3),"Precision:",round(TP/S2,3))

                
                
            return dice_scores,sens,precision 
        else:
            #Calculate dice coefficient for index pairs [index1, index2]

            dice_scores=np.zeros([num_labels+1,num_labels+1]) 
            for pair in indices_pairs: #Can use this to also create a sort of relative confusion matrix
                data_gt_i = data_gt == pair[1]
                data_pr_i = data_pr == pair[0]
                data_tp_i = (data_gt_i == data_pr_i)*data_gt_i 
                
                
                S1 = np.sum(data_gt_i)
                S2 = np.sum(data_pr_i)
                TP = np.sum(data_tp_i)

                dice_scores[pair[0],pair[1]]=calc_dice(TP,S1,S2)
                if(verbose):
                    print(f"Dice score for index {pair[0]} and {pair[1]} is {dice_scores[pair[0],pair[1]]}")
            return dice_scores
    else:
        print("Provided files are not of the same dimensions")
        print(data_gt.shape,data_pr.shape)

def calc_dice(TP,S1,S2):
    # TP = Number of true positive voxels
    # S1 = Number of voxels in segmentation 1 (GT)
    # S2 = Number of voxels in segmentation 2 (PR)
    return(2*TP/(S1+S2))

