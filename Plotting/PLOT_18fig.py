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
images = [19,24,11,43,2,28]
for i in images:
    MRI[i] = []
    MRI[i].append(mpimg.imread(r"C:\Users\halja\Desktop\MRI\\"+str(i)+".PNG"))
    MRI[i].append(mpimg.imread(r"C:\Users\halja\Desktop\MRI\\"+str(i)+"gt.PNG"))
    MRI[i].append(mpimg.imread(r"C:\Users\halja\Desktop\MRI\\"+str(i)+"seg.PNG"))


scalex = 0.7
scaley = 0.85



fig = plt.figure()
fig.tight_layout()
fig.set_size_inches(8.3*scalex,11.7*scaley)
outest = gridspec.GridSpec(6, 1, wspace=0.10, hspace=0.10) #width_ratios=[1.5,1]
 


for n,i in enumerate(images):
    outer = gridspec.GridSpecFromSubplotSpec(1, 3, wspace=0.1, hspace=0.0, subplot_spec=outest[n])
    for m in range(3):
        ax = plt.Subplot(fig, outer[m])
        ax.imshow(MRI[i][m],aspect='auto')

        ax.set_xticks([])
        ax.set_yticks([])

        if n == 0:
            if m == 0:
                ax.set_title(r"$\text{MRI Image}$", fontsize=10)
            if m == 1:
                ax.set_title(r"$\text{GT Segmentation}$", fontsize=10)
            if m == 2:
                ax.set_title(r"$\text{Auto Segmentation}$", fontsize=10)

        if m == 0:
            ax.set_ylabel(r"$\text{Image "+str(n+1)+r"}~$", fontsize=10, labelpad=10)
        fig.add_subplot(ax)


#fig.suptitle(r"$\textbf{Automatic Segmentations vs Ground Truth Segmentations of VTT}$",fontsize = 12)


plotname = str(random.randint(1, 10000))+".pdf"
print(plotname)
fig.savefig(plotname, bbox_inches='tight',dpi=400)

plt.show()
