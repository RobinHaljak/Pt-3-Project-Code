
import numpy as np
import matplotlib.pyplot as plt

TaskID = 510
fold = 0
T = 50

txt_filename = r"C:\Users\halja\Desktop\logs\\"+str(TaskID)+"_"+str(fold)+".txt"

f = open(txt_filename, "r")

data = f.read().split("\n")
print(data)
print(type(data))
print(len(data))
print(data[0])

train_losses = []
validation_losses = []
dice_scores = []
lr = []

for i in range(len(data)):
    if data[i][:5] == "epoch":
        #print("epoch",data[i][8:11])
        train_losses.append(data[i+1][41:48].replace("]",""))
        validation_losses.append(data[i+2][45:53].replace("]",""))
        dice_scores.append(data[i+3][61:67].replace("]",""))
        lr.append(data[i+5][32:40].replace("]",""))
        print(data[i+1][41:48],data[i+2][45:53],data[i+3][61:67],data[i+5][32:40])


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



train_losses_avg = moving_average(train_losses,n=T)
validation_losses_avg = moving_average(validation_losses,n=T)
dice_scores_avg = moving_average(dice_scores,n=T)


### oooh could keep track of these different moving averages and save models whenever the validation loss / dice starts growing and then compare the resulting models for different values of the moving average
### the positive - only need to do one run through and just continously save the models through it
###

plt.plot(train_losses_avg)
plt.plot(validation_losses_avg)
plt.show()

plt.plot(dice_scores_avg)
plt.show()