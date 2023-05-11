import nibabel as nib
import numpy as np
from os import listdir
from os.path import join

#Determine precisions, recalls, FPRs, dice scores for plotting of AUC or AUPRC.

#Folder with model segmentation (.npz)
model_segs = r"C:\Users\halja\Desktop\RESULTS\segmentation results\510\naive510\510_naive"
#Folder with ground truth segmentations (.nii.gz)
gt_segs = r"C:\Users\halja\Desktop\RESULTS\segmentation results\labels"

N1 = -30
N2 = 20
M = 30

thresh_range = [10**(x/M) for x in range(N1,N2)]


model_seg_files = [x for x in listdir(model_segs) if x[-4:] == ".npz"]
gt_seg_files =  [x for x in listdir(gt_segs) if x[-7:] == ".nii.gz"]



precisions = []
recalls = []
dices = []
FPRs = []
for thresh in thresh_range:
    temp_pres = []
    temp_recall = []
    temp_dice = []
    temp_FPR = []
    for n,fname in enumerate(model_seg_files):
        # Load files
        npz = np.load(join(model_segs,fname))
        gt = nib.load(join(gt_segs,gt_seg_files[n]))

        gt = gt.get_fdata()
        npz = npz["softmax"][1,:,:,:]
        npz = np.swapaxes(npz,0,2)


        pr = np.zeros_like(npz) 
        pr[npz >= thresh] = 1 # Segment voxels with softmax above variable threshold value
        pr = pr.astype(int)
        tp = np.equal(gt,pr)*gt # True postive
        tn = np.equal(gt,pr)*(np.ones_like(gt)-gt) # True negative
        S1 = np.sum(gt)
        S2 = np.sum(pr)
        TP = np.sum(tp)
        TN = np.sum(tn)
        #print(S2,TP)
        FP = S2-TP
        
        # Calculate segmentation accuracy metrics
        FPR = FP / (FP + TN)
        RECALL = np.nan_to_num(TP/S1)
        PRES = np.nan_to_num(TP/S2)
        DICE = np.nan_to_num(2*TP/(S1+S2))

        temp_pres.append(PRES)
        temp_recall.append(RECALL)
        temp_dice.append(DICE)
        temp_FPR.append(FPR)

    print(np.average(temp_dice),np.average(temp_pres),np.average(temp_recall),np.average(temp_FPR))

    # Average segmentation accuracy metrics for a certain threshold
    precisions.append(np.average(temp_pres))
    recalls.append(np.average(temp_recall))
    dices.append(np.average(temp_dice))
    FPRs.append(np.average(temp_FPR))

print(precisions)
print(recalls)
print(dices)
print(FPRs)