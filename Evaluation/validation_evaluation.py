from dice_calc import *
import csv
import statistics

def validation_eval(val_path,seg_path,AMOS2022=False,verbose_ = True):
    #seg_path -- path to directory of automatic segmentations
    #val_path -- path to directory of ground truth segmentations
    filenames = [x for x in os.listdir(seg_path) if x[-7:] == ".nii.gz"]
    num_files = len(filenames)

    scores = []
    sens_scores = []
    prec_scores = []
    for fname in filenames:
        file_gt = os.path.join(val_path,fname)
        file_pr = os.path.join(seg_path,fname)

        if(verbose_):
            print(f"Validation Dice scores for {fname}")
        x = dice_score(file_gt,file_pr,verbose=verbose_)
        scores.append(x[0])
        sens_scores.append(x[1])
        prec_scores.append(x[2])
    scores_np = np.array(scores)
    num_labels = len(scores[0])
    avg_scores = np.zeros(len(scores[0]))
    avg_scores = sum(scores)/num_files
    median_scores = []

 
    for i in range(1,num_labels):
        median_scores.append(statistics.median(scores_np[:,i]))
    min_scores = []
    for i in range(1,num_labels):
        min_scores.append(min(scores_np[:,i]))
    
    MSE = sum((scores - avg_scores)**2)
    STDEV = np.sqrt(MSE)/np.sqrt(num_files-1)

    if (AMOS2022):
        Organs = {"0": "background", "1": "spleen", "2": "right kidney", "3": "left kidney", "4": "gall bladder", "5": "esophagus", "6": "liver", "7": "stomach", "8": "aorta", "9": "postcava", "10": "pancreas", "11": "right adrenal gland", "12": "left adrenal gland", "13": "duodenum", "14": "bladder", "15": "prostate/uterus"}
        print("Average Dice scores by organ:")
        for i in range(1,num_labels):
            print(Organs[str(i)],": ",round(avg_scores[i],3)," +- ", round(STDEV[i],3))
        print("")
        print("Median Dice scores by organ:")
        for i in range(1,num_labels):
            print(Organs[str(i)],": ",round(median_scores[i-1],3))
        print("")
        print("Minimum Dice scores by organ:")
        for i in range(1,num_labels):
            print(Organs[str(i)],": ",round(min_scores[i-1],3))
    else:
        print("Average Dice scores by index:")
        for i in range(num_labels):
            print(i,": ",round(avg_scores[i],3)," +- ", round(STDEV[i],3))

    return scores_np,sens_scores,prec_scores

