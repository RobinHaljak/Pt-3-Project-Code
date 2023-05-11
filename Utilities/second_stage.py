# 1. take the output of early stopping ensembling (niigz) / the original image
# 2. recrop both image/mask with a buffer
# 3. reformat into region expansion input version
# 4. run second stage inferenc

from os import listdir
from os.path import join
from pathlib import Path
from Preprocessing.precropping import save_cropped    
import nibabel as nib
import numpy as np
from Evaluation.physics import bbox2_3D, resolution
from Inference_Code import run_my_inference
from batchgenerators.utilities.file_and_folder_operations import load_pickle
import torch
import matplotlib.pyplot as plt
from segmentation_and_postprocessing.postprocessing import postprocess
from evaluation.dice_calc import dice_score

def moving_average(arr, N):
    cumsum = np.cumsum(np.insert(arr, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def decaying_MA(arr,N):
    arr_len = len(arr)
    i=0
    MA = 0
    moving_averages = []
    for j in range(arr_len):

        #if j == 0:
            #MA = arr[0]
        #else:
        MA = MA*i/(i+1)+arr[j]/(i+1)
        
        moving_averages.append(MA)

        if j < N:
            i += 1

    return moving_averages

def closest_values(arr, best_epoch, N_best=1):
    # Calculate the absolute difference between each element of arr and best_epoch
    diff = np.abs(arr - best_epoch)
    
    # Find the indices of the N_best minimum differences
    idx = np.argpartition(diff, N_best)[:N_best]
    
    # Return the elements of arr with the N_best minimum differences
    return arr[idx]

def find_crossovers(arr1,arr2,crossover_type='both'):
    # assume len(arr1) <= len(arr2)
    #crossover point is the later point of the crossover
    crossovers = []
    if crossover_type == 'both':
        for i in range(len(arr1)-1):
            if ((arr1[i] > arr2[i]) and (arr1[i-1] < arr2[i-1])) or ((arr1[i] < arr2[i]) and (arr1[i-1] > arr2[i-1])):
                crossovers.append(i)
        return crossovers
    elif crossover_type == 'descending':
        for i in range(len(arr1)-1):
            if (arr1[i] < arr2[i]) and (arr1[i-1] > arr2[i-1]):
                crossovers.append(i)
        return crossovers
    elif crossover_type == 'ascending':
        for i in range(len(arr1)-1):
            if (arr1[i] > arr2[i]) and (arr1[i-1] < arr2[i-1]):
                crossovers.append(i)
        return crossovers
    else:
        print("Invalid crossover type")
        return None

def early_stopping(folds,path_to_results,MA_a = 25,MA_b= 50,mode="loss",N_best=1):
    best_checkpoints = {}
    for fold in folds:
        
        path_to_final_model = join(path_to_results,"fold_"+str(fold),"model_final_checkpoint.model")
        saved_model = torch.load(path_to_final_model,map_location=torch.device('cpu'))

        plotstuff = saved_model["plot_stuff"]
        val_loss = plotstuff[1]
        val_dice = plotstuff[3] 

        if mode == "loss":
            val_loss_MA_a = decaying_MA(val_loss,MA_a)
            val_loss_MA_b = decaying_MA(val_loss,MA_b)
            crossovers = find_crossovers(val_loss_MA_a,val_loss_MA_b,crossover_type='ascending')
        
            if len(crossovers) > 0:
                smallest_val_loss = 1000000
                best_epoch = 0
                for cross in crossovers:
                    if val_loss_MA_a[cross] < smallest_val_loss:
                        smallest_val_loss = val_loss_MA_a[cross]
                        best_epoch = cross
                
                if smallest_val_loss > val_loss_MA_a[-1]:
                    best_epoch = 750
            else:
                best_epoch = 750

        elif mode == "dice":
            val_dice_MA_a = decaying_MA(val_dice,MA_a)
            val_dice_MA_b = decaying_MA(val_dice,MA_b)
            crossovers = find_crossovers(val_dice_MA_a,val_dice_MA_b,crossover_type='descending')
        
            if len(crossovers) > 0:
                largest_val_dice = -1
                best_epoch = 0
                for cross in crossovers:
                    if val_dice_MA_a[cross] > largest_val_dice:
                        largest_val_dice = val_dice_MA_a[cross]
                        best_epoch = cross

                if largest_val_dice < val_dice_MA_a[-1]:
                    best_epoch = 750
            else:
                best_epoch = 750
        

        model_folder = join(path_to_results,"fold_"+str(fold))
        model_epochs = np.array([int(m[9:12]) for m in listdir(model_folder) if (m[:8] == "model_ep" and m[-4:] != ".pkl")])
        #print(model_epochs)
        best_available_epochs = closest_values(model_epochs,best_epoch,N_best=N_best)

        print("Fold",fold,"best model(s)",best_available_epochs)
        best_checkpoints[fold] = ["model_ep_%03d" % s for s in best_available_epochs]
    
    print(best_checkpoints)

    return best_checkpoints

if __name__ == "__main__":

    run_inference = False
    determine_early_stopping = True
    run_ensembling = True
    run_dice_evalutation = True
    #threshhold = 0.1

    MA_a = 25
    MA_b = 50

    power = 4

    run_name = '515'
    ensemble_name = "ensemble_testing"
    task_ID = 'Task510_NAXIVANEW'
    RE_task_ID = 'Task512_REGION2'

    network_type = '3d_fullres'
    models = ['nnUNetTrainerV2__nnUNetPlansv2.1',]
    folds = (0,1,2,3,4)
    delete_temp_files = False
    disable_postprocessing = True

    base_folder = "/home/rh756/rds/hpc-work/ANALYSIS"
    mask_input_folder_niigz = join(base_folder,task_ID,ensemble_name,"niigz")
    mask_input_folder_npz = join(base_folder,task_ID,ensemble_name,"npz")
    mask_files = sorted(listdir(mask_input_folder_npz))
    
    image_input_folder = join("/home/rh756/rds/hpc-work/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data",task_ID,"imagesTs")
    gt_segmentation_folder = join("/home/rh756/rds/hpc-work/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/",task_ID,"labelsTs")
    RE_input_folder = join(base_folder,task_ID,ensemble_name,"region_expansion","input")
    RE_output_folder = join(base_folder,task_ID,ensemble_name,"region_expansion","output")



    buffer = {}
    init_shapes = {}
    for n,mask_file in enumerate(mask_files):
        # open mask niigz
        print(join(mask_input_folder_npz,mask_file))
        
        mask = np.load(join(mask_input_folder_npz,mask_file),allow_pickle=True)
        #choose the softmax of the VTT
        mask1 = mask['softmax'][1,:,:,:] 
        print(mask1.shape)
        #choose the softmax of background
        mask0 = mask['softmax'][0,:,:,:] 

        # Normalise
        mask_adj= np.nan_to_num(mask1 / (mask1+mask0))
        mask_adj = np.swapaxes(mask_adj,0,2)
        mask_adj = mask_adj / np.amax(mask_adj) 

        im = nib.load(join(image_input_folder,mask_file[:-4]+".nii.gz"))
        img_data = im.get_fdata()
        img_affine = im.affine
        img_hdr = im.header
        img_shape = img_data.shape
        img_res = resolution(im)

        output_mask_borders = bbox2_3D(mask_adj)

        # Apply a buffer at the edges of the segmentation bounding box
        ratio = 19 # buffer ratio - width of buffer given by relevant dimension of image shape / ratio

        buffer_ = (int(output_mask_borders[0]-img_shape[0]//ratio),
                int(output_mask_borders[1]+img_shape[0]//ratio),
                int(output_mask_borders[2]-img_shape[1]//ratio),
                int(output_mask_borders[3]+img_shape[1]//ratio),
                int(output_mask_borders[4]-img_shape[2]//ratio),
                int(output_mask_borders[5]+img_shape[2]//ratio))
        
        buffer[n] = buffer_ 
        init_shapes[n] = img_shape

        print(buffer[n])
        
        out_fname_img = join(RE_input_folder,mask_file[:-5]+"0.nii.gz")
        out_fname_soft = join(RE_input_folder,mask_file[:-5]+"1.nii.gz")

        save_cropped(img_data,mask_adj,buffer,None,out_fname_img,out_fname_soft,img_affine,img_hdr,ID=n,mode='box')

    if determine_early_stopping:

        path_to_results = join("/home/rh756/rds/hpc-work/nnUNet/nnUNet_RESULTS_FOLDER/nnUNet/3d_fullres",
                                        RE_task_ID,models[0])
        best_checkpoints = early_stopping(folds,path_to_results)

    if run_inference:
        for fold in folds:
            for checkpoint in best_checkpoints[fold]:

                output_folder = join(RE_output_folder,"fold"+str(fold),checkpoint)
                Path(output_folder).mkdir(parents=True, exist_ok=True)

                print(RE_task_ID,checkpoint,(fold,),RE_input_folder,output_folder)
                run_my_inference(RE_task_ID,checkpoint,(fold,),RE_input_folder,output_folder,delete_temp_files,disable_postprocessing,network_type = network_type,models=models,num_modalities=2)

    #### Combine inferences from different folds:

    if run_ensembling:
        output_folder = join(RE_output_folder,"fold"+str(folds[0]),best_checkpoints[folds[0]][0],models[0])
        npz_out_files = [f for f in sorted(listdir(output_folder)) if f[-4:] == ".npz"]

        print(output_folder)
        print(npz_out_files)
        
        for n,filename in enumerate(npz_out_files):
            npz_total = None
            num_npzs = 0
            for fold in folds:
                for checkpoint in best_checkpoints[fold]:
                    out_file = join(RE_output_folder,"fold"+str(fold),best_checkpoints[fold][0],models[0],filename)
                    num_npzs += 1
                    if type(npz_total) != np.ndarray:
                        npz_total = np.load(out_file)["softmax"]
                    else:
                        npz_total += np.load(out_file)["softmax"]
            npz_total = npz_total / num_npzs
            RE_ensemble_out = join(RE_output_folder,"ensemble_npz",filename)
            RE_ensemble_out_niigz = join(RE_output_folder,"ensemble_niigz",filename[:-4]+".nii.gz")
            RE_ensemble_out_segmentation = join(RE_output_folder,"ensemble_segmentation",filename[:-4]+".nii.gz")

            print(init_shapes[n])
            npz_original = np.zeros((2,init_shapes[n][2],init_shapes[n][1],init_shapes[n][0]))
            print(buffer[n])
            npz_original[:,buffer[n][4]:buffer[n][5],buffer[n][2]:buffer[n][3],buffer[n][0]:buffer[n][1]] = npz_total #switched axes 1,3 to stay in npz standard being (z,y,x)
            print("Saving:",RE_ensemble_out)
            

            # Post-process 2nd stage segmentation

            segmentation, softmax, region_sizes = postprocess("",seg_threshhold=0.30,ROI_weight=1.50,post_process_ROIs=[1],softmax=npz_original,verbose=False,power=power,threshholded_largest_only=True,threshholded_largest_only_ratio=0.50)
            print(region_sizes)
            data = nib.load(join(mask_input_folder_niigz,filename[:-4]+".nii.gz"))

            img = data.get_fdata()
            hdr = data.header
            affine = data.affine

            img1 = nib.Nifti1Image(np.swapaxes(segmentation,0,2), affine,hdr)
            img2 = nib.Nifti1Image(np.swapaxes(softmax,0,2), affine,hdr)

            nib.save(img2, RE_ensemble_out_niigz) 
            nib.save(img1, RE_ensemble_out_segmentation) 
            np.savez(RE_ensemble_out,softmax=npz_original) #

    if run_dice_evalutation:
        out_segmentation_folder = join(RE_output_folder,"ensemble_segmentation")
        dice_scores = []
        sens_scores = []
        pres_scores = []
        for fname in sorted(listdir(out_segmentation_folder)):
            print(fname)
            dice_results = dice_score(join(gt_segmentation_folder,fname[:11]+fname[-7:]),join(out_segmentation_folder,fname))
            dice_scores.append(dice_results[0])
            sens_scores.append(dice_results[1])
            pres_scores.append(dice_results[2])
        print("Dices:",[i[0] for i in dice_scores])
        print("Average Dice:",np.average(dice_scores))

    