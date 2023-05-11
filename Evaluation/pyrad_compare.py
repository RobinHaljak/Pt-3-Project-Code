import os
import csv
import SimpleITK as sitk
import six
import numpy as np

from radiomics import featureextractor

def radiomic_eval(images_folder,gt_labels,seg_labels,ROI_index,extractor_settings,radiomic_features = "Default"):

    if radiomic_features == "Default":
        radiomic_features = ['diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass', 'diagnostics_Image-interpolated_Spacing', 'diagnostics_Image-interpolated_Size', 'diagnostics_Image-interpolated_Mean', 'diagnostics_Image-interpolated_Minimum', 'diagnostics_Image-interpolated_Maximum', 'diagnostics_Mask-interpolated_Spacing', 'diagnostics_Mask-interpolated_Size', 'diagnostics_Mask-interpolated_BoundingBox', 'diagnostics_Mask-interpolated_VoxelNum', 'diagnostics_Mask-interpolated_VolumeNum', 'diagnostics_Mask-interpolated_CenterOfMassIndex', 'diagnostics_Mask-interpolated_CenterOfMass', 'diagnostics_Mask-interpolated_Mean', 'diagnostics_Mask-interpolated_Minimum', 'diagnostics_Mask-interpolated_Maximum', 'original_shape_Elongation', 'original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume', 'original_shape_MinorAxisLength', 'original_shape_Sphericity', 'original_shape_SurfaceArea', 'original_shape_SurfaceVolumeRatio', 'original_shape_VoxelVolume', 'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength']

    num_features = len(radiomic_features)
    
    # Feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(extractor_settings)

    # File names
    im_files = os.listdir(images_folder)
    gt_files = os.listdir(gt_labels)
    seg_files = os.listdir(seg_labels)

    num_images = len(im_files)

    print(num_images,"validation images")
    print(num_features,"radiomic features")
    print("region of interest:",ROI_index)

    results_gt = []
    results_seg = []

    res_gt_avg = {}
    res_seg_avg = {}
    dif = {}

    for j in range(num_images): #Iterate over validation images
        im_file = os.path.join(images_folder,im_files[j])
        gt_file = os.path.join(gt_labels,gt_files[j])
        seg_file = os.path.join(seg_labels,seg_files[j])

        print(im_files[j],gt_files[j],seg_files[j])
        
        # Determine radiomic features
        results_gt.append(extractor.execute(im_file,gt_file,label=ROI_index))
        try:
            results_seg.append(extractor.execute(im_file,seg_file,label=ROI_index)) # can do this as part of next step without saving results like this
        except ValueError:
            results_seg.append(results_seg[-1])
            print(seg_file[j],"GIVING SAME RADIOMICS SINCE NO MASK")

    # Formatting for output
    for j in range(num_images):
        for key, val in six.iteritems(results_gt[j]):
            if key in radiomic_features:
                if key not in res_gt_avg:
                    res_gt_avg[key] = [val]
                else:
                    res_gt_avg[key].append(val)
        for key, val in six.iteritems(results_seg[j]):
            if key in radiomic_features:
                if key not in res_seg_avg:
                    res_seg_avg[key] = [val]
                else:
                    res_seg_avg[key].append(val)

    # Averaging radiomic feature values
    for key in radiomic_features:
        dif[key] = np.array(res_seg_avg[key]) - np.array(res_gt_avg[key])
    dif_avg = {}
    dif_stdev = {}
    for key in radiomic_features:
        dif_avg[key] = np.average(dif[key])
        dif_stdev[key] = np.std(dif[key])
        print(key,dif_avg[key],dif_stdev[key],print(float(dif_avg[key])/(float(dif_stdev[key])+0.0000001)))

    return res_gt_avg, res_seg_avg


        
