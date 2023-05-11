import random
import pandas as pd

# Create by-patient split of NAXIVA trial images into training / test set.

num = 45

ind = [i for i in range(num)]

random.shuffle(ind)

frac = 1/3

num_val = round(frac*num)

validation_indices = ind[:num_val]
validation_indices.sort()
print("Validation indices:",validation_indices)


print("Creating per patient split, accounting for different centres")

Patients = {}

Patients["IDs"] = [101,102,103,104,105,106,201,202,203,204,205,601,603,604,605,606,801,903,905]

print(len(Patients["IDs"]),"IDs")
#By centre:
# 1: 15
# 2: 12
# 6: 12
# 8: 3
# 9: 3

Patients["Categories"] = [1,1,1,1,1,1,2,2,2,2,2,6,6,6,6,6,8,9,9]

#Doing 75:25 -> want 5 validation, 14 training patients, Ideally have around 11-12 images in those validation cases
print()

df = pd.DataFrame(Patients)

print(df)

a = df.groupby('Categories',group_keys=False).apply(lambda x: x.sample(1))

print(a)

#104,201,603,801,905 -- 3+3+1+3+2

# --> ImageNo: [8,9,10,15,16,17,29,39,40,41,43,44] -> 12 validation images
# Can't really split validation set by centre -- just split by patient

TrainingSet = [101,102,103,105,106,202,203,204,205,601,604,605,606,903]


for i in range(4):
    print("Training Fold:",i)

    random.shuffle(TrainingSet)
    print(TrainingSet.pop())
    random.shuffle(TrainingSet)
    print(TrainingSet.pop())
    random.shuffle(TrainingSet)
    print(TrainingSet.pop())
    random.shuffle(TrainingSet)


# 0: 7
# 1: 7
# 2: 6
# 3: 7
# 4: 6


### Creating CHAOS split

CHAOS_IDs = [1,2,3,5,8,10,13,15,19,20,21,22,31,32,33,34,36,37,38,39]

print("CHAOS:",len(CHAOS_IDs),"IDs")

random.shuffle(CHAOS_IDs)
print("VALIDATION IDs:",CHAOS_IDs[:6])
