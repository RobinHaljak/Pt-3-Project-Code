import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import csv
import random

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts} \usepackage[T1]{fontenc} \usepackage{times} \usepackage{newtxmath}'



csv_file = r"C:\Users\halja\Desktop\MyProjectCode\pt-3-project\PLOT_DICEDIST.csv"

x_s = [] # labels of x-axis
y_s = []



title = r'$\textbf{Distribution of segmentation}~\mathbf{DICE}~\textbf{scores for 2-stage nnUNet}$'
legend = []
xlabel = r"$\text{Dice Score}$"
ylabel = r"$\text{Number of segmentations}$"


colors = ['k','b']
output_name = " testing123456789"
#check if file by this name already does exist, in this case add a letter or smthning

legend_fsize = 10
label_fsize = 15
title_fsize = 18

with open(csv_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i,row in enumerate(spamreader):
        if i == 0:
            series_names = []
        elif i == 1:
            num_series = len([j for j in row if j != ""]) // 1
            print(f"Plotting {num_series} series")
            for k in range(num_series):
                x_s.append([])
            for k in range(num_series):
                x_s[k].append(float(row[k]))
        else:
            for k in range(num_series):
                if row[k] != "":
                    x_s[k].append(float(row[k]))




fig, ax = plt.subplots()


for i in range(num_series):
    bins = [-0.05+x*0.1 for x in range(11)]
    ax.hist(x_s,bins=bins,stacked=True, density=False,color=["navy","darkgreen"],rwidth=0.62)

legend.append('Cross-Validation Segmentations')
legend.append('Test Set Segmentations')

ax.set_ylim(bottom = 0)
ax.set_xlim(left = -0.05,right = 1)
ax.legend(legend,fontsize=legend_fsize)
ax.set_xlabel(xlabel, fontsize=label_fsize)
ax.set_ylabel(ylabel, fontsize=label_fsize)
ax.set_title(title, fontsize=title_fsize)
fig.tight_layout()
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xticks(np.arange(0, 1, 0.1))

scalex = 0.46
scaley = 0.65
fig.set_size_inches(18.5*scalex, 10.5*scaley)


plotname = str(random.randint(1, 10000))+".pdf"
fig.savefig(plotname, bbox_inches='tight',dpi=100)
print(plotname)

plt.show()