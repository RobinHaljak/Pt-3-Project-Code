from batchgenerators.utilities.file_and_folder_operations import *

# Create custom NAXIVA splits (by patient)

fle = load_pickle("/home/rh756/rds/hpc-work/nnUNet/nnUNet_preprocessed/Task510_NAXIVANEW/splits_final.pkl")
print(fle)

print(fle[0]['val'])
split = {}
split[0] = [("NAXIVA_%04d") % k for k in [42,0,1,2,3,4,5]]
split[1] = [("NAXIVA_%04d") % k for k in [11,12,24,25,26,33,34]]
split[2] = [("NAXIVA_%04d") % k for k in [13,14,46,20,21,22,23]]
split[3] = [("NAXIVA_%04d") % k for k in [18,19,30,31,32,27,28]]
split[4] = [("NAXIVA_%04d") % k for k in [35,36,37,38,6,7,45]]


allids = []

for i in range(5):
    allids += split[i]
print(allids)

for i in range(5):
    curr_ids = []
    for j in range(5):
        if i != j:
            curr_ids += split[j]
    fle[i]['train'] = curr_ids
    fle[i]['val'] = split[i]

print(fle)
#fle[0]['train'] = ['amos_0585', 'amos_0582', 'amos_0586', 'amos_0508', 'amos_0540', 'amos_0514', 'amos_0541', 'amos_0592', 'amos_0578', 'amos_0532', 'amos_0588', 'amos_0530', 'amos_0594', 'amos_0600', 'amos_0584', 'amos_0571', 'amos_0507', 'amos_0599', 'amos_0548', 'amos_0587', 'amos_0510', 'amos_0555', 'amos_0596', 'amos_0518', 'amos_0538', 'amos_0517', 'amos_0595', 'amos_0580', 'amos_0589', 'amos_0570', 'amos_0557', 'amos_0583']
#fle[0]['val'] = ['amos_0558', 'amos_0590', 'amos_0591', 'amos_0554', 'amos_0597', 'amos_0593', 'amos_0522', 'amos_0551']

#fle[1]['train'] = ['amos_0584', 'amos_0507', 'amos_0599', 'amos_0532', 'amos_0538', 'amos_0595', 'amos_0551', 'amos_0591', 'amos_0583', 'amos_0586', 'amos_0597', 'amos_0555', 'amos_0590', 'amos_0540', 'amos_0589', 'amos_0554', 'amos_0548', 'amos_0570', 'amos_0518', 'amos_0578', 'amos_0588', 'amos_0558', 'amos_0530', 'amos_0522', 'amos_0510', 'amos_0587', 'amos_0594', 'amos_0593', 'amos_0580', 'amos_0514', 'amos_0571', 'amos_0592']
#fle[1]['val'] = ['amos_0557', 'amos_0517', 'amos_0600', 'amos_0585', 'amos_0541', 'amos_0508', 'amos_0596', 'amos_0582']

#fle[2]['train'] = ['amos_0540', 'amos_0570', 'amos_0555', 'amos_0586', 'amos_0597', 'amos_0583', 'amos_0522', 'amos_0593', 'amos_0587', 'amos_0551', 'amos_0600', 'amos_0580', 'amos_0530', 'amos_0588', 'amos_0554', 'amos_0541', 'amos_0508', 'amos_0591', 'amos_0514', 'amos_0585', 'amos_0596', 'amos_0594', 'amos_0582', 'amos_0599', 'amos_0590', 'amos_0538', 'amos_0595', 'amos_0518', 'amos_0548', 'amos_0558', 'amos_0557', 'amos_0517']
#fle[2]['val'] = ['amos_0532', 'amos_0578', 'amos_0592', 'amos_0510', 'amos_0571', 'amos_0584', 'amos_0589', 'amos_0507']

#fle[3]['train'] = ['amos_0517', 'amos_0587', 'amos_0599', 'amos_0594', 'amos_0596', 'amos_0578', 'amos_0538', 'amos_0522', 'amos_0593', 'amos_0580', 'amos_0551', 'amos_0592', 'amos_0554', 'amos_0590', 'amos_0584', 'amos_0510', 'amos_0591', 'amos_0508', 'amos_0597', 'amos_0586', 'amos_0558', 'amos_0557', 'amos_0582', 'amos_0595', 'amos_0541', 'amos_0540', 'amos_0532', 'amos_0507', 'amos_0585', 'amos_0600', 'amos_0589', 'amos_0571']
#fle[3]['val'] = ['amos_0530', 'amos_0588', 'amos_0518', 'amos_0583', 'amos_0548', 'amos_0555', 'amos_0514', 'amos_0570']

#fle[4]['train'] = ['amos_0589', 'amos_0588', 'amos_0514', 'amos_0555', 'amos_0518', 'amos_0591', 'amos_0510', 'amos_0517', 'amos_0570', 'amos_0507', 'amos_0584', 'amos_0592', 'amos_0530', 'amos_0532', 'amos_0582', 'amos_0554', 'amos_0557', 'amos_0596', 'amos_0590', 'amos_0571', 'amos_0583', 'amos_0508', 'amos_0585', 'amos_0578', 'amos_0600', 'amos_0522', 'amos_0541', 'amos_0548', 'amos_0551', 'amos_0593', 'amos_0558', 'amos_0597']
#fle[4]['val'] = ['amos_0595', 'amos_0540', 'amos_0538', 'amos_0599', 'amos_0580', 'amos_0587', 'amos_0594', 'amos_0586']


write_pickle(fle,"/home/rh756/rds/hpc-work/nnUNet/nnUNet_preprocessed/Task510_NAXIVANEW/splits_final.pkl")

fle = load_pickle("/home/rh756/rds/hpc-work/nnUNet/nnUNet_preprocessed/Task510_NAXIVANEW/splits_final.pkl")

print(fle)