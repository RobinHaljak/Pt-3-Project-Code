from Inference_Code import run_my_inference
from evaluation.dice_calc import dice_score
from segmentation_and_postprocessing.postprocessing import postprocess
from Utilities.FOV_cropping import determine_empty_bounds
from second_stage import early_stopping
from os import listdir, mkdir
from os.path import isfile, join, isdir
import numpy as np
import pickle
import json
import nibabel as nib
import matplotlib.pyplot as plt



def crop_segmentation_using_image(seg_data,image_data):


    #Axis 0
    rowsums = np.sum(image_data,axis=(1,2))
    init_sum = np.sum(seg_data)
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    seg_data = seg_data[crop_index_L:crop_index_R,:,:]

    #Axis 1
    rowsums = np.sum(image_data,axis=(0,2))
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    seg_data = seg_data[:,crop_index_L:crop_index_R,:]

    #Axis 2
    rowsums = np.sum(image_data,axis=(0,1))
    crop_index_L , crop_index_R = determine_empty_bounds(rowsums)
    seg_data = seg_data[:,:,crop_index_L:crop_index_R]

    final_sum = np.sum(seg_data)

    print("Cropped shape:",seg_data.shape)
    print(init_sum,final_sum)
    assert final_sum == init_sum, "These really should be the same!!!"


    return seg_data

    
def determine_early_stops(output_folders,gt_segmentation_folder,N=5):

    RES = {}
    # Determine dice score of segmentations by each checkpoint model
    for out in output_folders:
        onlyfiles = sorted([f for f in listdir(out) if isfile(join(out, f))])
        scores = []
        print(" ")
        print(out[-12:])


        for filename in onlyfiles:
            scores.append(dice_score(join(gt_segmentation_folder,filename[:11]+filename[16:]),join(out,filename),verbose=False))

        print("avg:",np.round(np.average(scores),3))
        RES[out[-12:]] = np.average(scores)
    
    ### Find 5 best early stopping models for each
    best_models = {}

    d_scores = list(RES.values())
    d_scores_sorted = sorted(d_scores,reverse=True)
    N_best = d_scores_sorted[:N]
    best_models = [checkpoint_names[d_scores.index(f)] for f in N_best]
    print("Best models:")
    for i in range(N):
        print(best_models[i],RES[best_models[i]])

    return best_models, RES


def finetune_hyperparams(output_folder_base,filenames,seg_threshholds,ROI_weights,best_models,N=5):

    RES = {}
    for seg in seg_threshholds:
        RES[seg] = {}
        for weight in ROI_weights:
            RES[seg][weight] = []

    for fname in filenames:
        npz_data = None
        for model in best_models[fold]:
            npz_path = output_folder_base+'fold'+str(fold)+"_val/"+model+"/nnUNetTrainerV2__nnUNetPlansv2.1/"+fname[:-7]+".npz"
            npz_data_full = np.load(npz_path)

            if type(npz_data) != np.ndarray: #not the cleanest coding I've ever done lol
                npz_data = npz_data_full["softmax"].astype(np.float32)
            else:
                npz_data += npz_data_full["softmax"].astype(np.float32)

        npz_data = npz_data / N  #Ensembled softmax
        
        for seg in seg_threshholds:
            for weight in ROI_weights:

                segmentation, softmax, region_sizes = postprocess("",seg_threshhold=seg,ROI_weight=weight,post_process_ROIs=[1],softmax=npz_data)
                segmentation = np.swapaxes(segmentation,0,2) # npzs have axis ordering (z,y,x) ; niftis have (x,y,z) iirc...
                print(fname,seg,weight)

                data_gt = nib.load(join(gt_segmentation_folder,fname[:11]+fname[16:]))
                seg_gt = data_gt.get_fdata()

                data = nib.load(join(input_folder,fname))
                img_gt = data.get_fdata()

                seg_gt = crop_segmentation_using_image(seg_gt,img_gt)

                dice_out, x, y = dice_score("","",data_pr=segmentation,data_gt=seg_gt)
                if np.isnan(dice_out[0]):
                    dice_out = 0
                RES[seg][weight].append(float(dice_out[0]))

    return RES


def final_model_ensembling(seg_final,ROI_final,best_hyperparams,best_models,output_folder_base,output_folder_test,input_folder_test,N_best=5,folds=(0,1,2,3,4),thresh_final=0.5,verbose=False):
    softmax_total = None
    for fold in folds:
        npz_data = None
        for model in best_models[fold]:
            npz_path = output_folder_base+'fold'+str(fold)+"_test/"+model+"/nnUNetTrainerV2__nnUNetPlansv2.1/"+fname[:-7]+".npz"
            npz_data_full = np.load(npz_path)

            if type(npz_data) != np.ndarray: #not the cleanest coding I've ever done lol
                npz_data = npz_data_full["softmax"].astype(np.float32)
            else:
                npz_data += npz_data_full["softmax"].astype(np.float32)

        npz_data = npz_data / N_best  #Ensembled softmax for fname

        print("SOFTMAX MAX:",np.amax(npz_data))
        segmentation, softmax, region_sizes = postprocess("",seg_threshhold=best_hyperparams[0],ROI_weight=best_hyperparams[1],post_process_ROIs=[1],softmax=npz_data,threshholded_largest_only=True,threshholded_largest_only_ratio=thresh_final,verbose=verbose)

        if type(softmax_total) != np.ndarray: #not the cleanest coding I've ever done lol
            softmax_total = softmax
        else:
            softmax_total += softmax
        

    softmax_total / N_best

    segmentation, softmax, region_sizes = postprocess("",seg_threshhold=seg_final,ROI_weight=ROI_final,post_process_ROIs=[1],softmax=softmax_total,verbose=verbose)
    #Load reference image:

    ref_path = join(input_folder_test,fname)
    new_path_niigz = join(output_folder_test,"niigz",fname)
    if not isdir(join(output_folder_test,"niigz")):
        mkdir(join(output_folder_test,"niigz"))

    image_name = fname.replace(".nii.gz",".npz")
    new_path_npz = join(output_folder_test,"npz",image_name)
    if not isdir(join(output_folder_test,"npz")):
        mkdir(join(output_folder_test,"npz"))

    data = nib.load(ref_path)

    img = data.get_fdata()
    hdr = data.header
    affine = data.affine

    img = np.swapaxes(segmentation,0,2)
    

    img1 = nib.Nifti1Image(img, affine,hdr)

    nib.save(img1, new_path_niigz) 

    np.savez(new_path_npz,softmax=softmax)







if __name__ == "__main__":

    run_name = '510'
    ensemble_name = "ensemble"
    task_ID = 'Task510_NAXIVANEW'
    folds = (0,1,2,3,4) # Inference folds

    # Values of post-processing hyperparameters for grid search
    seg_threshholds = [0.010,0.030,0.050,0.100,0.200]
    ROI_weights = [0.50,0.75,1.00,1.25,1.50,2.00,3.00]

    N_best = 5 # How many checkpoints to take per fold

    #lengths of early stopping
    conventional_early_stopping = False
    MA_a = 25 
    MA_b = 50

    # Can skip inference tasks if already run once (or have relevant files saved)
    do_validation_inference = False
    do_finetuning = True
    do_test_set_inference = False
    do_test_set_ensembling = True

    do_pres_sens_curve = False
    ROI_weight_curve = [np.exp(i/3) for i in range(-3,10)]

    # Final post-processing & segmentation hyperparameters.

    seg_final = 0.01 # seg_threshhold for the final postprocessing (was 0.10 earlier, changed for ROC analysis)
    ROI_final = 1 # ROI_weight for the final postprocessing
    thresh_final = 0.99 # volume threshholds for smaller subregions

    input_folder_base = '/home/rh756/rds/hpc-work/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task510_NAXIVANEW/'
    output_folder_base = '/home/rh756/rds/hpc-work/ANALYSIS/'+task_ID+'/'
    gt_segmentation_folder = "/home/rh756/rds/hpc-work/nnUNet/nnUNet_preprocessed/Task510_NAXIVANEW/gt_segmentations"
    input_folder_test = "/home/rh756/rds/hpc-work/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task510_NAXIVANEW/imagesTs" #oops
    gt_label_folder = '/home/rh756/rds/hpc-work/NAXIVA_NEW/labelsTr'

    network_type = '3d_fullres'
    models = ['nnUNetTrainerV2__nnUNetPlansv2.1',]

    # Initial and steps for the saved model checkpoint epochs
    model_init = 10
    model_final = 750
    model_step = 40

    delete_temp_files = False
    disable_postprocessing = False

    RESULTS = {}
    PP_RESULTS = {}
    best_models = {}

    for fold in folds:

        print("Fold:",fold)
        input_folder = input_folder_base + 'fold'+str(fold)+'_val' # input folder shoud be the validation set of the fold we're currently looking at
        output_base = output_folder_base + 'fold'+str(fold)+'_val/'
        print(input_folder)

        checkpoint_names = [("model_ep_%03d" % i) for i in range(model_init,model_final,model_step)]
        output_folders = [((output_base+"model_ep_%03d") % i) for i in range(model_init,model_final,model_step)]
        
        print(checkpoint_names)
        print(output_folders)

        ### RUN INFERENCE WITH DESIRED CHECKPOINT MODELS
        if do_validation_inference:
            if not conventional_early_stopping:
                for k in range(len(checkpoint_names)):
                    run_my_inference(task_ID,checkpoint_names[k],(fold,),input_folder,output_folders[k],delete_temp_files,disable_postprocessing,network_type = network_type,models=models)

        # Determine best checkpoints and their best hyperparameters
        if do_finetuning:
            if conventional_early_stopping:
                path_to_results = join("/home/rh756/rds/hpc-work/nnUNet/nnUNet_RESULTS_FOLDER/nnUNet/3d_fullres",task_ID,models[0])
                best_models = early_stopping(folds,path_to_results,N_best=N_best) #I'm doing this 5x instead of once but at this point I'm not going to fix it since it's quick anyway
                RESULTS = {"conventional ES":"nothing to put here :)"}

                conv_es_checkpoints = best_models[fold]
                conv_es_outfolders = [(output_base+x) for x in conv_es_checkpoints]
                if do_validation_inference:
                    for k in range(len(conv_es_checkpoints)):
                        run_my_inference(task_ID,conv_es_checkpoints[k],(fold,),input_folder,conv_es_outfolders[k],delete_temp_files,disable_postprocessing,network_type = network_type,models=models)
            else:
                #1. Determine best checkpoints
                best_models[fold], RESULTS[fold] = determine_early_stops(output_folders,gt_segmentation_folder,N=N_best)

            #2. Fine-tune best checkpoint ensemble post-processing hyperparameters

            filenames = sorted([f for f in listdir(output_folders[0]) if isfile(join(output_folders[0], f))]) # the way I handle all of this in finetune_hyperparams is some ugly coding, but for now - we move
            PP_RESULTS[int(fold)] = finetune_hyperparams(output_folder_base,filenames,seg_threshholds,ROI_weights,best_models,N=N_best)


    #3. Save results (or open if using pre-done results)

    if do_finetuning:
        print(RESULTS)
        PP_RESULTS["best_models"] = list(best_models)

        with open(output_folder_base+run_name+"_"+ensemble_name+"_RES.json", 'w') as f: 
            json.dump(RESULTS, f)   

        with open(output_folder_base+run_name+"_"+ensemble_name+"_RES_PP.json", 'w') as f: 
            json.dump(PP_RESULTS, f)

    else:
        with open(output_folder_base+run_name+"_"+ensemble_name+"_RES_PP.json", 'r') as f: 
            PP_RESULTS = json.load(f)
        
        best_models = PP_RESULTS["best_models"]
        for i in range(5):
            best_models[i] = best_models[str(i)] #Not sure why reading from a json messes this up but we move
        for fold in folds:
            PP_RESULTS[fold] = {}
            for seg in seg_threshholds:
                PP_RESULTS[fold][seg] = {}
                for weight in ROI_weights:
                    PP_RESULTS[fold][seg][weight] = PP_RESULTS[str(fold)][str(seg)][str(weight)]

    # 1. do inference of each fold (-> each best_checkpoint of it) on test set separately
    # 2. combine softmax results within each fold
    # 3. post-process each fold according to best hyperparameters
    # 4. combine each 
    # 5. run simple post-processing (i.e. single region, ROI_weight = 1, seg_threshhold = 0.03 + 2nd stage)


    delete_temp_files = False
    disable_postprocessing = False 

    network_type = '3d_fullres'
    models = ['nnUNetTrainerV2__nnUNetPlansv2.1',]
    

    test_filenames = sorted([f for f in listdir(input_folder_test) if isfile(join(input_folder_test, f))])

    output_folder_test = join(output_folder_base,ensemble_name)
    if not isdir(output_folder_test):
        mkdir(output_folder_test)

    ###### prediction
    
    for fold in folds:
        checkpoints = best_models[fold]
        best_dice = 0
        for seg in seg_threshholds:
            for weight in ROI_weights:
                if np.average(PP_RESULTS[fold][seg][weight]) > best_dice:
                    best_hyperparams = [seg,weight]
                    best_dice = np.average(PP_RESULTS[fold][seg][weight]) 
        print("Fold:",fold,"Models:",checkpoints,"Hyperparams:",best_hyperparams)

        
        output_base = output_folder_base + 'fold'+str(fold)+'_test/'

        checkpoint_names = checkpoints
        output_folders = [join(output_base,name) for name in checkpoint_names]
        
        print(checkpoint_names)
        print(output_folders)

        if do_test_set_inference:
            for k in range(len(checkpoint_names)):
                run_my_inference(task_ID,checkpoint_names[k],(fold,),input_folder_test,output_folders[k],delete_temp_files,disable_postprocessing,network_type = network_type,models=models)
    
    if do_test_set_ensembling:
        for fname in test_filenames:
            print("Final ensembling:")
            print(fname)
            final_model_ensembling(seg_final,ROI_final,best_hyperparams,best_models,output_folder_base,output_folder_test,input_folder_test,N_best=N_best,folds=folds,thresh_final=thresh_final)

    ### Evaluate dice scores of model segmentations
    
    dice_scores = []
    sens_scores = []
    pres_scores = []
    for fname in test_filenames:
        print(fname)
        dice_results = dice_score(join(gt_label_folder,fname),join(output_folder_test,"niigz",fname))
        dice_scores.append(dice_results[0])
        sens_scores.append(dice_results[1])
        pres_scores.append(dice_results[2])
    print("Dices:",[i[0] for i in dice_scores])
    print("Average Dice:",np.average(dice_scores))


        
















