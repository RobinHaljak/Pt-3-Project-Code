from batchgenerators.utilities.file_and_folder_operations import *

# Edit the nnUNet default splits to my by-patient splits, for region expansion training

path = "/home/rh756/rds/hpc-work/nnUNet/nnUNet_preprocessed/Task512_REGION2/splits_final.pkl"

fle = load_pickle(path)
print(fle)

region_ranges = [[0, 38], [38, 65], [65, 89], [89, 116], [116, 137], [137, 159], [159, 175], [175, 190], [190, 209], [209, 226], [226, 241], [241, 257], [257, 273], [273, 288], [288, 320], [320, 354], [354, 389], [389, 401], [401, 425], [425, 444], [444, 457], [457, 461], [461, 466], [466, 489], [489, 517], [517, 547], [547, 568], [568, 591], [591, 613], [613, 633], [633, 649], [649, 663], [663, 689], [689, 702], [702, 725]]
training_IDs = sorted([42,0,1,2,3,4,5,11,12,24,25,26,33,34,13,14,46,20,21,22,23,18,19,30,31,32,27,28,35,36,37,38,6,7,45])


print(len(region_ranges))
print(fle[0]['val'])
split = {}

a1 = [42,0,1,2,3,4,5]
a2 = [11,12,24,25,26,33,34]
a3 = [13,14,46,20,21,22,23]
a4 = [18,19,30,31,32,27,28]
a5 = [35,36,37,38,6,7,45]

b1 = []
b2 = []
b3 = []
b4 = []
b5 = []

for i in range(7):
    for j in range(region_ranges[training_IDs.index(a1[i])][0],region_ranges[training_IDs.index(a1[i])][1]):
        b1.append(j)
    for j in range(region_ranges[training_IDs.index(a2[i])][0],region_ranges[training_IDs.index(a2[i])][1]):
        b2.append(j)
    for j in range(region_ranges[training_IDs.index(a3[i])][0],region_ranges[training_IDs.index(a3[i])][1]):
        b3.append(j)
    for j in range(region_ranges[training_IDs.index(a4[i])][0],region_ranges[training_IDs.index(a4[i])][1]):
        b4.append(j)
    for j in range(region_ranges[training_IDs.index(a5[i])][0],region_ranges[training_IDs.index(a5[i])][1]):
        b5.append(j)

# Uninclude broken segmentations
split[0] = [("REGION_%04d") % k for k in b1 if k not in [369,371,383]]
split[1] = [("REGION_%04d") % k for k in b2 if k not in [369,371,383]]
split[2] = [("REGION_%04d") % k for k in b3 if k not in [369,371,383]]
split[3] = [("REGION_%04d") % k for k in b4 if k not in [369,371,383]]
split[4] = [("REGION_%04d") % k for k in b5 if k not in [369,371,383]]

print(split[0])
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


write_pickle(fle,path)

fle = load_pickle(path)

print(fle)