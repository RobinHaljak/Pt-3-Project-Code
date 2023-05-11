import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import csv
import random

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts} \usepackage[T1]{fontenc} \usepackage{times} \usepackage{newtxmath}'



csv_file = r"C:\Users\halja\Desktop\MyProjectCode\pt-3-project\PLOT_EB1.csv"

x_s = []
x_s1 = []
x_s2 = []
y_s = []
y_s1 = []
y_s2 = []


#title = r'$\textbf{Model}~\mathbf{vs.}~\textbf{Ground truth VTT Volume}~\textit{(mm}^\mathbf{3}\textit{)}$'
#xlabel = r"$\text{Ground truth segmentation VTT Volume}~\textit{(mm}^\mathbf{3}\textit{)}$"
#ylabel = r"$\text{Model segmentation VTT Volume}~\textit{(mm}^\mathbf{3}\textit{)}$"


title = r'$\textbf{Model}~\mathbf{vs.}~\textbf{Ground truth VTT length}~\textit{(mm)}$'
legend = []
xlabel = r"$\text{Ground truth segmentation VTT length}~\textit{(mm)}$"
ylabel = r"$\text{Model segmentation VTT length}~\textit{(mm)}$"


colors = ['k','b']
output_name = " testing123456789"
#check if file by this name already does exist, in this case add a letter or smthning

legend_fsize = 10
label_fsize = 15
title_fsize = 18

trendline = False

x_y_line = True

with open(csv_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i,row in enumerate(spamreader):
        if i == 0:
            series_names = []
        elif i == 1:
            num_series = len([j for j in row if j != ""]) // 6
            print(f"Plotting {num_series} series")
            for k in range(num_series):
                x_s.append([])
                y_s.append([])
                x_s1.append([])
                y_s1.append([])
                x_s2.append([])
                y_s2.append([])
            for k in range(num_series):
                x_s[k].append(float(row[6*k]))
                x_s1[k].append(float(row[6*k+1]))
                x_s2[k].append(float(row[6*k+2]))
                y_s[k].append(float(row[6*k+3]))
                y_s1[k].append(float(row[6*k+4]))
                y_s2[k].append(float(row[6*k+5]))
        else:
            for k in range(num_series):
                if row[6*k] != "":
                    x_s[k].append(float(row[6*k]))
                    x_s1[k].append(float(row[6*k+1]))
                    x_s2[k].append(float(row[6*k+2]))
                    y_s[k].append(float(row[6*k+3]))
                    y_s1[k].append(float(row[6*k+4]))
                    y_s2[k].append(float(row[6*k+5]))


fig, ax = plt.subplots()


for i in range(num_series):

    x = np.array(x_s[i])
    x1 = np.array(x_s1[i])
    x2 = np.array(x_s2[i])

    y = np.array(y_s[i])
    y1 = np.array(y_s1[i])
    y2 = np.array(y_s2[i])



    print(x)
    print(y)
    ax.errorbar(x, y,xerr=[x1,x2],yerr=[y1,y2], c=colors[i], capsize=3,ls='none') #marker='+'

if trendline:
    legend.append("Best linear fit")
    x = np.concatenate(x_s)
    y = np.concatenate(y_s)
    slope, intercept = np.polyfit(x, y, 1)
    x_min = min(x) - abs(0.50*min(x))
    x_max = max(x) + abs(0.05*max(x))
    x_trendline = np.array([x_min,x_max])
    y_trendline = slope*x_trendline + intercept
    ax.plot(x_trendline, y_trendline, color='green', linestyle='--', linewidth = 0.75)

if x_y_line:
    x = np.concatenate(x_s)
    legend.append("True length")
    x_min = 0
    x_max = max(x) + abs(0.05*max(x))
    x_trendline = np.array([x_min,x_max])
    y_trendline = x_trendline
    ax.plot(x_trendline, y_trendline, color='red', linestyle='--', linewidth = 0.75)

legend.append('Validation set VTTs')
legend.append('Test set VTTs')

#ax.set_ylim(bottom = 0)
#ax.set_xlim(left = 0)
ax.legend(legend,fontsize=legend_fsize)
ax.set_xlabel(xlabel, fontsize=label_fsize)
ax.set_ylabel(ylabel, fontsize=label_fsize)
ax.set_title(title, fontsize=title_fsize)

#ax.set_xscale("log")
#ax.set_yscale("log")

ax.tick_params(axis='both', which='major', labelsize=14)

scalex = 0.45
scaley = 0.75
fig.set_size_inches(18.5*scalex, 10.5*scaley)


plotname = str(random.randint(1, 10000))+".pdf"
fig.savefig(plotname, bbox_inches='tight',dpi=100)
print(plotname)


plt.show()