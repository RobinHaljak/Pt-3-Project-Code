from os import listdir
from os.path import join
### Complete analsis pipeline for segmentations

# Step 1: Calculate dice scores
# Step 2: Calculate radiomic features for both GT and SEG
# Step 3: Save per ROI in csv files: each validation file separated
import numpy as np
import csv
from validation_evaluation import validation_eval
from dice_calc import dice_score
from pyrad_compare import radiomic_eval
import matplotlib.pyplot as plt
from segmentation_and_postprocessing.postprocessing import postprocess

# Full analysis (dice scores & radiomics comparison) of segmentations in two folders.

images_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results\test_images"
gt_segmentations_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results\test_labels"
model_segmentations_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results\510\naive510\510naivetest"
output_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results"
# REQUIRES corresponding images in gt_segmentations and model_segmenations to have the same name

radiomics_settings = "C:\\Users\\halja\\Desktop\\MyProjectCode\\pt-3-project\\dev\\pyradiomics\\test_params.yml"
radiomic_features = "Default" # Replace with list of radiomic features if wish to specify, by default includes all quantifiable features

if radiomic_features == "Default": # Default = ALL features
    radiomic_features = ['diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass', 'diagnostics_Image-interpolated_Spacing', 'diagnostics_Image-interpolated_Size', 'diagnostics_Image-interpolated_Mean', 'diagnostics_Image-interpolated_Minimum', 'diagnostics_Image-interpolated_Maximum', 'diagnostics_Mask-interpolated_Spacing', 'diagnostics_Mask-interpolated_Size', 'diagnostics_Mask-interpolated_BoundingBox', 'diagnostics_Mask-interpolated_VoxelNum', 'diagnostics_Mask-interpolated_VolumeNum', 'diagnostics_Mask-interpolated_CenterOfMassIndex', 'diagnostics_Mask-interpolated_CenterOfMass', 'diagnostics_Mask-interpolated_Mean', 'diagnostics_Mask-interpolated_Minimum', 'diagnostics_Mask-interpolated_Maximum', 'original_shape_Elongation', 'original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume', 'original_shape_MinorAxisLength', 'original_shape_Sphericity', 'original_shape_SurfaceArea', 'original_shape_SurfaceVolumeRatio', 'original_shape_VoxelVolume', 'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength']

run_name = "515_TEST"
ROI_indices = [1]

num_rad_features = len(radiomic_features)

im_files = [x for x in listdir(images_folder) if x[-7:] == ".nii.gz"]
num_im_files = len(im_files)

seg_files = [x for x in listdir(model_segmentations_folder) if x[-7:] == ".nii.gz"]
num_seg_files = len(seg_files)

# Initialise np array to be converted to csv:

output_array = np.zeros([2+num_rad_features,1+2*num_im_files])

#Calculate dice scores
dice_scores,sens_scores_,pres_scores_ = validation_eval(gt_segmentations_folder,model_segmentations_folder,AMOS2022=False)

dice_output = ["DICE"]

#Calculate radiomics for each index

for ROI_index in ROI_indices:

    formatted_im_files = [ROI_index]
    for x in im_files:
        formatted_im_files.append(x.replace(".nii.gz","")+"_GT")
        formatted_im_files.append(x.replace(".nii.gz","")+"_SEG")

    output_array = [formatted_im_files]

    # Save dice scores
    for i in range(num_seg_files):
        dice_output.append(1)
        dice_output.append(dice_scores[i][0])
    output_array.append(dice_output)

    # Determine radiomic features
    radiomics_gt, radiomics_seg = radiomic_eval(images_folder,gt_segmentations_folder,model_segmentations_folder,ROI_index,radiomics_settings,radiomic_features = radiomic_features)

    # Reformat radiomics results
    for key in radiomic_features:
        key_out = [key]
        for i in range(num_im_files):
            if isinstance(radiomics_gt[key][i],tuple):
                key_out.append(radiomics_gt[key][i][0])
                key_out.append(radiomics_seg[key][i][0])
            elif(isinstance(radiomics_gt[key][i],np.ndarray)):
                key_out.append(float(radiomics_gt[key][i]))
                key_out.append(float(radiomics_seg[key][i]))
            else:
                key_out.append(radiomics_gt[key][i])
                key_out.append(radiomics_seg[key][i])
        output_array.append(key_out)

    ### Create csvs 
    print("SAVING TO:")
    np.savetxt(join(output_folder,run_name+"_"+str(ROI_index)+".csv"), 
        output_array,
        delimiter =", ", 
        fmt ='% s')
    
    for i in range(num_seg_files):
        print(dice_scores[i][0])
    print(" ")
    for i in range(num_seg_files):
        print(sens_scores_[i])
    print(" ")
    for i in range(num_seg_files):
        print(pres_scores_[i])



