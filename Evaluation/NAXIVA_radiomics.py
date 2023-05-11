from os import listdir
from os.path import join
import six
import numpy as np
from radiomics import featureextractor

# Determine the image radiomics for the NAXIVA data set.


def evaluate_and_save(images_folder,labels_folder,output_folder,ROI_indices,radiomics_settings,radiomic_features = "Default",use_index_mapping = False):

    if radiomic_features == "Default":
        radiomic_features = ['diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass', 'diagnostics_Image-interpolated_Spacing', 'diagnostics_Image-interpolated_Size', 'diagnostics_Image-interpolated_Mean', 'diagnostics_Image-interpolated_Minimum', 'diagnostics_Image-interpolated_Maximum', 'diagnostics_Mask-interpolated_Spacing', 'diagnostics_Mask-interpolated_Size', 'diagnostics_Mask-interpolated_BoundingBox', 'diagnostics_Mask-interpolated_VoxelNum', 'diagnostics_Mask-interpolated_VolumeNum', 'diagnostics_Mask-interpolated_CenterOfMassIndex', 'diagnostics_Mask-interpolated_CenterOfMass', 'diagnostics_Mask-interpolated_Mean', 'diagnostics_Mask-interpolated_Minimum', 'diagnostics_Mask-interpolated_Maximum', 'original_shape_Elongation', 'original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume', 'original_shape_MinorAxisLength', 'original_shape_Sphericity', 'original_shape_SurfaceArea', 'original_shape_SurfaceVolumeRatio', 'original_shape_VoxelVolume', 'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength']

    im_files = sorted(listdir(images_folder))
    num_im_files = len(im_files)

    for ROI_index in ROI_indices:

        # Format image filenames for .csv
        formatted_im_files = ["ROI INDEX: "+str(ROI_index)]
        for x in im_files:
            if use_index_mapping:
                formatted_im_files.append(index_mapping[x])
            else:
                formatted_im_files.append(x.replace(".nii.gz","")+"_GT")
        output_array = [formatted_im_files]

        # Calculate radiomic features
        radiomics = radiomic_eval_single(images_folder,labels_folder,ROI_index,radiomics_settings,radiomic_features)

        # Extracting values of radiomic features for output
        for key in radiomic_features:
            key_out = [key]
            for i in range(num_im_files):
                # Dealing with different datatypes, radiomic features with multiple output values per image 
                # (e.g. centre of mass) not supported for now (will only output first value of array)
                if isinstance(radiomics[key][i],tuple):
                    key_out.append(radiomics[key][i][0])
                elif(isinstance(radiomics[key][i],np.ndarray)):
                    key_out.append(float(radiomics[key][i]))
                else:
                    key_out.append(radiomics[key][i])
            output_array.append(key_out)

        ### Create .csvs 
        np.savetxt(join(output_folder,run_name+"_"+str(ROI_index)+".csv"), 
            output_array,
            delimiter =", ", 
            fmt ='% s')


def radiomic_eval_single(images_folder,labels_folder,ROI_index,extractor_settings,radiomic_features):

    #pyradiomics extractor for feature extraction
    extractor = featureextractor.RadiomicsFeatureExtractor(extractor_settings)

    im_files = sorted(listdir(images_folder))
    seg_files = sorted(listdir(labels_folder))
    num_features = len(radiomic_features)
    num_images = len(im_files)

    print(num_images,"validation images")
    print(num_features,"radiomic features")
    print("region of interest:",ROI_index)

    results_seg = []
    res_seg = {}

    for j in range(num_images): #Iterate over validation images
        im_file = join(images_folder,im_files[j])
        seg_file = join(labels_folder,seg_files[j])
        if(im_files[j]==seg_files[j]):
            print(im_files[j])
        else:
            #print("WARNING -- NON_MATCHING NAMES: You may have used the wrong folders.")
            print("Image:",im_files[j],"Segmentation:",seg_files[j])
        try:
            radiomics_output = extractor.execute(im_file,seg_file,label=ROI_index)
            results_seg.append(radiomics_output)

        except: # This is just here because I don't have time to fix it properly for when there is no segmentation in image...
            results_seg.append(radiomics_output)
            print("No seg for file, replace radiomics with last file radiomics")
    for j in range(num_images): #calculate averages and standard deviations of radiomic feature values
        for key, val in six.iteritems(results_seg[j]):
            if key in radiomic_features:
                if key not in res_seg:
                    res_seg[key] = [val]
                else:
                    res_seg[key].append(val)

    for key in radiomic_features:
        print(key,np.array(res_seg[key]))

    return res_seg

if __name__ == '__main__':

    output_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results"

    #Input folders of nifti files, requires files to be sorted in same order in both
    images_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results\images"
    #labels_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results\labels"
    labels_folder = r"C:\Users\halja\Desktop\RESULTS\segmentation results\1stagelabels"
    #indices of all ROIs to be processed:
    ROI_indices = [1]

    #Path to .yaml settings file, see: https://pyradiomics.readthedocs.io/en/latest/customization.html#radiomics-parameter-file-label
    radiomics_settings = "C:\\Users\\halja\\Desktop\\MyProjectCode\\pt-3-project\\dev\\pyradiomics\\test_params.yml"
    
    #Determines name of output .csv
    run_name = "baseline510"

    #List of radiomic features to be evaluated:
    radiomic_features = ['original_shape_Elongation', 'original_shape_Flatness', 
                         'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 
                         'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 
                         'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 
                         'original_shape_MeshVolume', 'original_shape_MinorAxisLength', 
                         'original_shape_Sphericity', 'original_shape_SurfaceArea', 'original_shape_SurfaceVolumeRatio', 
                         'original_shape_VoxelVolume', 'original_firstorder_10Percentile', 
                         'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 
                         'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 
                         'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 
                         'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 
                         'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 
                         'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 
                         'original_firstorder_Uniformity', 'original_firstorder_Variance']
    # set: radiomic_features = "Default" to evaluate all quantifiable features.
    # see:https://pyradiomics.readthedocs.io/en/latest/features.html
    # (always evaluates all features under the hood since I am lazy)
    radiomic_features = "Default"
    #Ignore
    use_index_mapping = False
    index_mapping = { #Not completely accurate!
        "NAXIVA_0000_0000.nii.gz":"N0101_0",
        "NAXIVA_0001_0000.nii.gz":"N0101_1",
        "NAXIVA_0002_0000.nii.gz":"N0101_2",
        "NAXIVA_0003_0000.nii.gz":"N0102_0",
        "NAXIVA_0004_0000.nii.gz":"N0102_1",
        "NAXIVA_0005_0000.nii.gz":"N0102_2",
        "NAXIVA_0006_0000.nii.gz":"N0103_0",
        "NAXIVA_0007_0000.nii.gz":"N0103_1",
        "NAXIVA_0008_0000.nii.gz":"N0104_0",
        "NAXIVA_0009_0000.nii.gz":"N0104_1",
        "NAXIVA_0010_0000.nii.gz":"N0104_2",
        "NAXIVA_0011_0000.nii.gz":"N0105_0",
        "NAXIVA_0012_0000.nii.gz":"N0105_1",
        "NAXIVA_0013_0000.nii.gz":"N0106_1",
        "NAXIVA_0014_0000.nii.gz":"N0106_2",
        "NAXIVA_0015_0000.nii.gz":"N0201_0",
        "NAXIVA_0016_0000.nii.gz":"N0201_1",
        "NAXIVA_0017_0000.nii.gz":"N0201_2",
        "NAXIVA_0018_0000.nii.gz":"N0202_0",
        "NAXIVA_0019_0000.nii.gz":"N0202_1",
        "NAXIVA_0020_0000.nii.gz":"N0203_0",
        "NAXIVA_0021_0000.nii.gz":"N0203_1",
        "NAXIVA_0022_0000.nii.gz":"N0203_2",
        "NAXIVA_0023_0000.nii.gz":"N0204_0",
        "NAXIVA_0024_0000.nii.gz":"N0205_0",
        "NAXIVA_0025_0000.nii.gz":"N0205_1",
        "NAXIVA_0026_0000.nii.gz":"N0205_2",
        "NAXIVA_0027_0000.nii.gz":"N0601_0",
        "NAXIVA_0028_0000.nii.gz":"N0601_1",
        "NAXIVA_0029_0000.nii.gz":"N0603_0",
        "NAXIVA_0030_0000.nii.gz":"N0604_0",
        "NAXIVA_0031_0000.nii.gz":"N0604_1",
        "NAXIVA_0032_0000.nii.gz":"N0604_2",
        "NAXIVA_0033_0000.nii.gz":"N0605_0",
        "NAXIVA_0034_0000.nii.gz":"N0605_1",
        "NAXIVA_0035_0000.nii.gz":"N0606_0",
        "NAXIVA_0036_0000.nii.gz":"N0606_1",
        "NAXIVA_0037_0000.nii.gz":"N0606_2",
        "NAXIVA_0038_0000.nii.gz":"N0606_3",
        "NAXIVA_0039_0000.nii.gz":"N0801_0",
        "NAXIVA_0040_0000.nii.gz":"N0801_1",
        "NAXIVA_0041_0000.nii.gz":"N0801_2",
        "NAXIVA_0042_0000.nii.gz":"N0903_0",
        "NAXIVA_0043_0000.nii.gz":"N0905_0",
        "NAXIVA_0044_0000.nii.gz":"N0905_1",
        "NAXIVA_0045_0000.nii.gz":"N0103_2",
        "NAXIVA_0046_0000.nii.gz":"N0106_0",
    }

    evaluate_and_save(images_folder,labels_folder,output_folder,ROI_indices,radiomics_settings,radiomic_features = radiomic_features)





