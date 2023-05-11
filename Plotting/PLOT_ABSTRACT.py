import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import csv
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts} \usepackage[T1]{fontenc} \usepackage{times} \usepackage{newtxmath}'



MRI = {}
MRI[0] = mpimg.imread(r"C:\Users\halja\Desktop\MRIs for abstract\Image0001.PNG")[0:838,0:1037,:]
MRI[1] = mpimg.imread(r"C:\Users\halja\Desktop\MRIs for abstract\Image0016.PNG")[0:838,0:1037,:]
MRI[2] = mpimg.imread(r"C:\Users\halja\Desktop\MRIs for abstract\Image0032.PNG")[0:838,0:1037,:]
SEG = {}
SEG[0] = mpimg.imread(r"C:\Users\halja\Desktop\MRIs for abstract\Seg0001.PNG")[0:838,0:1037,:]
SEG[1] = mpimg.imread(r"C:\Users\halja\Desktop\MRIs for abstract\Seg0016.PNG")[0:838,0:1037,:]
SEG[2] = mpimg.imread(r"C:\Users\halja\Desktop\MRIs for abstract\Seg0032.PNG")[0:838,0:1037,:]

INF = mpimg.imread(r"C:\Users\halja\Desktop\MRIs for abstract\inf.png")

for j in range(3):
    print(MRI[j].shape)

scalex = 1
scaley = 0.8


fig = plt.figure()
fig.set_size_inches(18.5*scalex, 10.5*scaley)
outest = gridspec.GridSpec(1, 2, wspace=0.19, hspace=0.00,width_ratios=[1.5,1])

outer = gridspec.GridSpecFromSubplotSpec(3, 1,subplot_spec=outest[0], wspace=0, hspace=0.10)

inner0 = gridspec.GridSpecFromSubplotSpec(1, 1,
                    subplot_spec=outer[0], wspace=0, hspace=0)

inner1 = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[1], wspace=0.03, hspace=0)
inner2 = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[2], wspace=0.03, hspace=0)

ax = plt.Subplot(fig, inner0[0])
#t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (1, j))
#t.set_ha('center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r"$\mathbf{a)}$", loc='left', fontsize=12)
ax.imshow(INF)

fig.add_subplot(ax)

for j in range(3):
    if j == 1:
        ax.set_title(r"$\mathbf{c)}$", loc='left', fontsize=12)
    ax = plt.Subplot(fig, inner1[j])
    #t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (1, j))
    #t.set_ha('center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(MRI[j])
    fig.add_subplot(ax)

for j in range(3):
    if j == 1:
        ax.set_title(r"$\mathbf{d)}$", loc='left', fontsize=12)
    ax = plt.Subplot(fig, inner2[j])
    #t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (1, 1))
    #t.set_ha('center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(SEG[j])
    fig.add_subplot(ax)

outer = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outest[1], wspace=0, hspace=0)

inner1 = gridspec.GridSpecFromSubplotSpec(1, 1,
                    subplot_spec=outer[0], wspace=0, hspace=0)






csv_file = r"C:\Users\halja\Desktop\MyProjectCode\pt-3-project\PLOT_3VAR.csv"

x_s = []
y_s = []
c_s = []

title = r'$\mathbf{ROI}~\mathbf{Voxels}~\mathrm{vs.}~\textbf{Average Edge Contrast}~\text{for 2-stage model.}$'
legend = ['2-stage model',"Test Dice scores"]  
ylabel = r"$\mathbf{ROI}~\mathbf{Voxels}$"
xlabel = r"$\mathbf{Average~Edge~Contrast}$"
ColormapLabel = r"$\text{Dice score}$"

style = ['k+','bx']
output_name = " testing123456789"
#check if file by this name already does exist, in this case add a letter or smthning

legend_fsize = 10
label_fsize = 12
title_fsize = 14

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

ax = plt.Subplot(fig, inner1[0])

norm = mcolors.TwoSlopeNorm(vcenter=0.4, vmin=0, vmax=1)

for i in range(num_series):

    x = np.array(x_s[i])
    y = np.array(y_s[i])
    c = np.array(c_s[i])
    
    scatter = ax.scatter(x, y, c = c ,marker = 'o',cmap=my_cmap)

ax.legend(legend,fontsize=legend_fsize)
ax.set_xlabel(xlabel, fontsize=label_fsize)
ax.set_ylabel(ylabel, fontsize=label_fsize)
#ax.set_title(title, fontsize=title_fsize)
#ax.set_ylim(bottom = -750)
ax.tick_params(axis='both', which='major', labelsize=14)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='10%', pad=0.15)

plt.colorbar(scatter, ax=ax, cax=cax)

cax.set_ylabel(ColormapLabel, rotation=270,labelpad= 10,fontsize=label_fsize)

#t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (1, 1))
#t.set_ha('center')

ax.set_title(r"$\mathbf{b)}$", loc='left', fontsize=12)
ax.tick_params(axis='y', rotation=70,labelsize=8)
ax.tick_params(axis='y',labelsize=8)
#ax.set_xticklabels(ax.get_xticks(), rotation = 45)

fig.add_subplot(ax)

plotname = str(random.randint(1, 10000))+".png"
print(plotname)
fig.savefig(plotname, bbox_inches='tight',dpi=1200)


fig.show()

plt.show()

