
from os import listdir
from os.path import join
from dicom_to_nifti_my_ver import *

#Main folder for CHAOS MRI images


def convert_with_name(input_path,output_base,im_index,modality):

    output_name = "CHAOS_"+modality+"_"+im_index
    output_folder = output_base+modality
    output_path = output_folder+"\\"+output_name+".nii.gz"
    convert(input_path,output_path)



base_folder = "C:\\Users\\halja\\Desktop\\MyProjectCode\Chaos\\Train_Sets\\MR\\"

output_base = "C:\\Users\\halja\\Desktop\\MyProjectCode\\CHAOS_Converted\\"

im_indices = listdir(base_folder)


for im in im_indices:
    #1. T1DUAL
    input_path_T1_i = base_folder+im+"\\T1DUAL\\DICOM_anon\\InPhase"
    convert_with_name(input_path_T1_i,output_base,im,"T1DUAL_IN")

    input_path_T1_o = base_folder+im+"\\T1DUAL\\DICOM_anon\\OutPhase"
    convert_with_name(input_path_T1_o,output_base,im,"T1DUAL_OUT")

    #2. T2SPIR
    input_path_T2 = base_folder+im+"\\T2SPIR\\DICOM_anon"
    convert_with_name(input_path_T2,output_base,im,"T2SPIR")

