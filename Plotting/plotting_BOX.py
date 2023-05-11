import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import csv
import random

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts} \usepackage[T1]{fontenc} \usepackage{times} \usepackage{newtxmath}'



csv_file = r"C:\Users\halja\Desktop\MyProjectCode\pt-3-project\PLOT_BOX.csv"

x_s = []
y_s = []

title = r'$\textbf{Differences in VTT characteristics for segmented and unsegmented VTTs}$'
legend = ['Validation Dice scores',"Test Dice scores"]  
xlabel = r"$\text{VTT Volume}~\textit{(mm}^\mathbf{3}\textit{)}$"
ylabel = [r"$\mathbf{VTT}~\mathbf{Volume}~\textit{(mm}^\mathbf{3}\textit{)}$",
          r"$\mathbf{VTT}~\mathbf{Length}~\textit{(mm)}$",
          r"$\mathbf{Voxel}~\mathbf{Volume}~\textit{(mm}^\mathbf{3}\textit{)}$",
          r"$\mathbf{Average~Edge~Contrast}$",
          r"$\mathbf{SNR}$"
          ]

style = ['k+','bx']
output_name = " testing123456789"
#check if file by this name already does exist, in this case add a letter or smthning

legend_fsize = 10
label_fsize = 12
title_fsize = 20

trendline = False

with open(csv_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i,row in enumerate(spamreader):
        if i == 0:
            series_names = []
        elif i == 1:
            num_series = len([j for j in row if j != ""]) //2
            print(f"Plotting {num_series} series")
            for k in range(num_series):
                x_s.append([])
                y_s.append([])
            for k in range(num_series):
                x_s[k].append(float(row[2*k]))
                y_s[k].append(float(row[2*k+1]))
        else:
            for k in range(num_series):
                if row[2*k] != "":
                    x_s[k].append(float(row[2*k]))
                if row[2*k+1] != "":
                    y_s[k].append(float(row[2*k+1]))


fig, ax = plt.subplots(ncols=5, figsize=(12, 5))

#ncols=4, figsize=(12, 5), sharey=False



for i in range(num_series):

    print(i)
    x = np.array(x_s[i])
    y = np.array(y_s[i])


    print(x)
    print(y)
    box = ax[i].boxplot([x],positions = [1],patch_artist=True,boxprops = dict(facecolor = "darkgreen", linewidth = 1.5),medianprops = dict(color = "deepskyblue", linewidth = 2.0))
    box = ax[i].boxplot([y],positions = [2],patch_artist=True,boxprops = dict(facecolor = "firebrick", linewidth = 1.5),medianprops = dict(color = "deepskyblue", linewidth = 2.0))

    #plt.setp(box['boxes'],color = "Green")
    

if trendline:
    legend.append("Best linear fit")
    
    x = np.concatenate(x_s)
    y = np.concatenate(y_s)
    slope, intercept = np.polyfit(x, y, 1)
    x_min = min(x) - abs(0.10*min(x))
    x_max = max(x) - abs(0.05*max(x))
    x_trendline = np.array([x_min,x_max])
    y_trendline = slope*x_trendline + intercept
    ax.plot(x_trendline, y_trendline, color='red', linestyle='--', linewidth = 0.75)



fig.suptitle(title, fontsize=title_fsize)
for i in range(num_series):
    ax[i].set_xticklabels(["Segmented","Unsegmented"],rotation=15,fontsize=6)
    #ax[i].set_ylabel(ylabel[i], fontsize=label_fsize)
    ax[i].set_title(ylabel[i], fontsize=14)

    ax[i].tick_params(axis='both', which='major', labelsize=13)
    

#ax.set_xscale("log")

scalex = 0.55
scaley = 0.70
fig.set_size_inches(18.5*scalex, 10.5*scaley)


plotname = str(random.randint(1, 10000))+".pdf"
plt.tight_layout()
fig.savefig(plotname, bbox_inches='tight',dpi=100)

print(plotname)

plt.show()