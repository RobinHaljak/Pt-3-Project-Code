import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts} \usepackage[T1]{fontenc} \usepackage{times} \usepackage{newtxmath}'



csv_file = r"C:\Users\halja\Desktop\MyProjectCode\pt-3-project\PLOT_3VAR.csv"

x_s = []
y_s = []
c_s = []

title = r'$\mathbf{Average~Edge~Contrast}~\mathrm{vs.}~\mathbf{Surface~Volume~Ratio}~\text{for 2-stage model.}$'
legend = ['2-stage model',"Test Dice scores"]  
ylabel = r"$\text{Average Edge Contrast}$"
xlabel = r"$\text{Surface-Volume ratio}$"
ColormapLabel = r"$\text{Dice score}$"

style = ['k+','bx']
output_name = " testing123456789"
#check if file by this name already does exist, in this case add a letter or smthning

legend_fsize = 10
label_fsize = 15
title_fsize = 18

trendline = False


a = 1.0

# Get the colormap colors, multiply them with the factor "a", and create new colormap
my_cmap = plt.cm.RdYlGn(np.arange(plt.cm.RdYlGn.N))

lightness_adj = np.array([[1-(128-abs(i-128))/700,1-(128-abs(i-128))/700,1-(128-abs(i-128))/700] for i in range(256)])
print(len(my_cmap))
print(type(my_cmap))
my_cmap[:,0:3] *= lightness_adj
my_cmap = mcolors.ListedColormap(my_cmap)

with open(csv_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i,row in enumerate(spamreader):
        if i == 0:
            series_names = []
        elif i == 1:
            num_series = len([j for j in row if j != ""]) // 3
            print(f"Plotting {num_series} series")
            for k in range(num_series):
                x_s.append([])
                y_s.append([])
                c_s.append([])
            for k in range(num_series):
                x_s[k].append(float(row[3*k]))
                y_s[k].append(float(row[3*k+1]))
                c_s[k].append(float(row[3*k+2]))
        else:
            for k in range(num_series):
                if row[2*k] != "":
                    x_s[k].append(float(row[3*k]))
                    y_s[k].append(float(row[3*k+1]))
                    c_s[k].append(float(row[3*k+2]))


fig, ax = plt.subplots()

norm = mcolors.TwoSlopeNorm(vcenter=0.4, vmin=0, vmax=1)

for i in range(num_series):

    x = np.array(x_s[i])
    y = np.array(y_s[i])
    c = np.array(c_s[i])
    
    scatter = ax.scatter(x, y, c = c ,marker = 'o',cmap=my_cmap)

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
#ax.set_ylim(bottom = -750)
ax.tick_params(axis='both', which='major', labelsize=14)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='10%', pad=0.2)

plt.colorbar(scatter, ax=ax, cax=cax)

cax.set_ylabel(ColormapLabel, rotation=270,labelpad= 14,fontsize=label_fsize)


plt.show()