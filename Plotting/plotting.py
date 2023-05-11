import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import csv
import random

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts} \usepackage[T1]{fontenc} \usepackage{times} \usepackage{newtxmath}'



csv_file = r"C:\Users\halja\Desktop\MyProjectCode\pt-3-project\PLOT_THIS.csv"

x_s = []
y_s = []

title = r'$\mathbf{Dice}~\mathrm{vs.}~\mathbf{Voxel}~\mathbf{Volume}~\textit{(mm}^\mathbf{3}\textit{)}~\text{for 2-stage model.}$'
legend = ['Validation Dice scores',"Test Dice scores"]  
xlabel = r"$\text{Voxel Volume}~\textit{(mm}^\mathbf{3}\textit{)}$"
ylabel = r"$\text{Dice score}$"

style = ['k+','bx']
output_name = " testing123456789"
#check if file by this name already does exist, in this case add a letter or smthning

legend_fsize = 10
label_fsize = 15
title_fsize = 18

trendline = True

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
                    y_s[k].append(float(row[2*k+1]))


fig, ax = plt.subplots()


for i in range(num_series):

    x = np.array(x_s[i])
    y = np.array(y_s[i])


    print(x)
    print(y)
    ax.plot(x, y, style[i])

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

ax.legend(legend,fontsize=legend_fsize)
ax.set_xlabel(xlabel, fontsize=label_fsize)
ax.set_ylabel(ylabel, fontsize=label_fsize)
ax.set_title(title, fontsize=title_fsize)

ax.tick_params(axis='both', which='major', labelsize=14)



scalex = 0.45
scaley = 0.50
fig.set_size_inches(18.5*scalex, 10.5*scaley)


plotname = str(random.randint(1, 10000))+".pdf"
fig.savefig(plotname, bbox_inches='tight',dpi=100)
print(plotname)

plt.show()