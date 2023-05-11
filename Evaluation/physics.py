import numpy as np
import nibabel as nib
from segmentation_and_postprocessing.dilate_or_erode import dilate,erode
from scipy.ndimage import convolve
from scipy.signal import correlate
from scipy.fft import fft
from scipy.stats import shapiro
from scipy.stats import kstest
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import json

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts} \usepackage[T1]{fontenc} \usepackage{times} \usepackage{newtxmath}'










### Metrics:

# 1. Spatial resolution 
# 2. Contrast: Weber Contrast: I_2 - I_1 / I_1  (I_1 is background (lower))
# 3. Noise: Standard deviation of uniform region of intensity within image (shot noise - large N -> gaussian noise: using standard deviation sensible)
# 4. Signal-to-Noise Ratio (SNR)
# 5. Contrast-to-Noise Ratio
# 6. Wiener spectrum: Also from noise segmentations
# 7. Rose model 
# 8. Edge contrast: find contrast between edge ROIs -> CNR (as opposed to CNR by just averaging the intensity values of the ROI)
# (9.) Contrast of VTT to kidneys, liver, spleen - see if any interesting results - this has some relevancce to the contrast agent concentrations etc in the body


# Find the bounding box for a volume in a 3d matrix.
# taken from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax




########## 1. Spatial resolution  ##########

def resolution(img):

    affine = img.affine
    x = np.sqrt(affine[0,0]**2+affine[1,0]**2+affine[2,0]**2)
    y = np.sqrt(affine[0,1]**2+affine[1,1]**2+affine[2,1]**2)
    z = np.sqrt(affine[0,2]**2+affine[1,2]**2+affine[2,2]**2)
    return [x,y,z]


########## 2. Average intensity and Standard Deviation of intensity of signal in ROI  ############

def avg_intensity(img_data,seg_data,ROI=1):

    seg = seg_data == ROI
    num_voxels = np.sum(seg)
    ROI_img = img_data * seg
    intensity_sum = np.sum(ROI_img)

    return intensity_sum / num_voxels

def stdev(img_data,seg_data,ROI=1):

    seg = seg_data == ROI
    ROI_img = img_data * seg

    ROI_img_values = ROI_img.flatten()
    ROI_img_values = ROI_img_values[ROI_img_values != 0]
    standard_deviation = np.std(ROI_img)

    return standard_deviation

########## 3. Contrast (Weber)  ############

def weber_contrast(img_data,seg_data,ROI=1,k=1):

    seg = seg_data == ROI

    avg_seg = avg_intensity(img_data,seg_data)
    seg_dil = dilate(seg,k)                                 

    edge_seg = seg_dil ^ seg #XORS 
    avg_edge = avg_intensity(img_data,edge_seg) # Average intensity of voxels around the edge of the ROI

    weber_contrast = abs((avg_seg-avg_edge)/np.min([avg_seg,avg_edge]))
    return weber_contrast
    # This is not the best way to find contrast, since one end of ROI could be embedded in material with higher signal, the other end in material with lower signal, and we end up with no contrast even though for a large region of the boundary it could be very well defined.
    # Hence want to use edge contrast


########## 4. Noise ############

# Getting noise segmentations in 3D Slicer
# 1. set brightness so 0 values get cut off, but you can clearly see the grey in non-zero empty space
# 2. register atleast 1000 voxels within 2 different empty space regions if possible
# 3. do segment statistics - record std deviation --> noise (check that estimate of noise doesn't vary too much between two "empty" regions, then average)

# This gives noise, CNR, SNR - also save segmentations of empty signal --> Wiener specctrum

def noise(noise_seg_data,image_data):

    noise_seg_data = noise_seg_data.flatten()
    image_data = image_data.flatten()
    noise_data = image_data[np.where(noise_seg_data==1)]

    standard_deviation = np.std(noise_data) # Take noise value to be standard deviation of the background signal

    return standard_deviation

# Plot noise spectrum
def noise_spectrum(noise_seg_data,image_data):
    noise_seg_data = noise_seg_data.flatten()
    image_data = image_data.flatten()
    noise_data = image_data[np.where(noise_seg_data==1)]

    title = r'$\textbf{Spectrum of background noise in MRI image}~\mathbf{(\sigma = 9.31)}$'
    legend = ['Noise intensity']
    xlabel = r"$\text{Noise Signal Intensity}$"
    ylabel = r"$\text{Voxel count}$"

    legend_fsize = 10
    label_fsize = 15
    title_fsize = 18

    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel, fontsize=label_fsize)
    ax.set_ylabel(ylabel, fontsize=label_fsize)
    ax.set_title(title, fontsize=title_fsize)

    ax.hist(noise_data,bins=int(np.amax(noise_data))+1,range=(0,int(np.amax(noise_data))+1),color="black")
    ax.legend(legend,fontsize=legend_fsize)

    scalex = 0.46
    scaley = 0.55
    fig.set_size_inches(18.5*scalex, 10.5*scaley)

    plotname = str(random.randint(1, 10000))+".pdf"
    fig.savefig(plotname, bbox_inches='tight',dpi=100)

    plt.show()




######### 5. Signal-to-Noise Ratio (SNR) ############

def signal_noise_ratio(image_data,seg_data,noise_seg_data,ROI=1):

    avg_signal = avg_intensity(image_data,seg_data,ROI=ROI)
    img_noise = noise(noise_seg_data,image_data)

    print("Signal-to-Noise ratio:",avg_signal / img_noise)
    return avg_signal / img_noise

######### 6. Contrast-to-Noise Ratio (CNR)

def contrast_noise_ratio(image_data,seg_data,noise_seg_data,ROI=1,k=1):
    
    contrast_avg = edge_contrast(image_data,seg_data,ROI=ROI,k=k)
    img_noise = noise(noise_seg_data,image_data)

    print("Contrast-to-Noise ratio:",contrast_avg / img_noise)
    return contrast_avg / img_noise


######### 7. Wiener spectrum  ###########

def wiener_spectrum(noise_seg_data,image_data,image_resolution):

    # 1. acquire 3d cube of iamge values within noise region
    
    noise_data = noise_seg_data*image_data

    # 1.5 Find bounding box

    bounding_box = bbox2_3D(noise_data)
    print(bounding_box)
    noise_data = noise_data[bounding_box[0]:bounding_box[1],bounding_box[2]:bounding_box[3],bounding_box[4]:bounding_box[5]]

    # 2. Subtract average signal from each pixel
    
    noise_data_flat = noise_data.flatten()
    noise_values = noise_data_flat[np.where(noise_data_flat!=0)]
    avg_noise = np.average(noise_values)
    
    # 3. Calculate autocorrelation as function of position

    # 4. FT of autocorrelation

    plot_things = False
    size = 256

    # Do separately along each axis:

    # X-AXIS
    spectrum_sum = np.zeros([size])
    for i in range(bounding_box[3]-bounding_box[2]):
        for j in range(bounding_box[5]-bounding_box[4]):
            x_0 = noise_data[:,i,j]
            if np.sum(x_0) != 0:
                non_zero = np.where(x_0!=0)
                non_zero_bounds = [np.amin(non_zero), np.amax(non_zero)]

                x_0 = x_0[non_zero_bounds[0]:non_zero_bounds[1]]
                x_0 = x_0 - np.average(x_0)

                if np.isnan(x_0).any():
                    continue

                
                straight_fft = np.fft.fft(x_0,size) # Fourier transform
                pwr = np.abs(straight_fft)**2 #  Get the Noise Power Spectrum
                frequencies = np.fft.fftfreq(size, 1)

                idx = np.argsort(frequencies)
                spectrum_sum += pwr[idx]

                if plot_things:
                    plt.plot(frequencies[idx],pwr[idx])
                    plt.show()
            else:
                continue
    if plot_things:
        plt.plot(frequencies[idx],spectrum_sum)
        plt.show()

    # Y-AXIS
    spectrum_sum = np.zeros([size])
    for i in range(bounding_box[1]-bounding_box[0]):
        for j in range(bounding_box[5]-bounding_box[4]):
            x_0 = noise_data[i,:,j]
            if np.sum(x_0) != 0:
                non_zero = np.where(x_0!=0)
                non_zero_bounds = [np.amin(non_zero), np.amax(non_zero)]

                x_0 = x_0[non_zero_bounds[0]:non_zero_bounds[1]]
                x_0 = x_0 - np.average(x_0)

                if plot_things:
                    plt.plot(x_0)
                    plt.show()
                    plt.close()

                if np.isnan(x_0).any():
                    continue

                straight_fft = np.fft.fft(x_0,size)
                pwr = np.abs(straight_fft)**2
                frequencies = np.fft.fftfreq(size, 1)

                idx = np.argsort(frequencies)
                spectrum_sum += pwr[idx]

                if plot_things:
                    plt.plot(frequencies[idx],pwr[idx])
                    plt.show()
            else:
                continue

    if plot_things:
        plt.plot(frequencies[idx],spectrum_sum)
        plt.show()

    # Z-AXIS
    spectrum_sum = np.zeros([size])
    for i in range(bounding_box[1]-bounding_box[0]):
        for j in range(bounding_box[3]-bounding_box[2]):

            x_0 = noise_data[i,j,:]
            if np.sum(x_0) != 0:
                non_zero = np.where(x_0!=0)
                non_zero_bounds = [np.amin(non_zero), np.amax(non_zero)]

                x_0 = x_0[non_zero_bounds[0]:non_zero_bounds[1]]
                x_0 = x_0 - np.average(x_0)

                if np.isnan(x_0).any():
                    continue


                straight_fft = np.fft.fft(x_0,size)
                pwr = np.abs(straight_fft)**2
                frequencies = np.fft.fftfreq(size, 1)

                idx = np.argsort(frequencies)
                spectrum_sum += pwr[idx]

                if plot_things:
                    plt.plot(frequencies[idx],pwr[idx])
                    plt.show()
            else:
                continue

    if plot_things:
        plt.plot(frequencies[idx],spectrum_sum)
        plt.show()

    return


######### 8. Rose model  #################

def rose_model(image_data,seg_data,ROI=1,k=1):

    # Is there any point to this? this is basically assuming that the ROI and background are uniform, which they are not...

    # 1. determine background (surrounding) signal (B)
    # 2. determine (edge) contrast (C)
    # 3. determine number of voxels in ROI (S)

    background = (dilate(seg_data,3) - seg_data)*image_data

    bg_flat = background.flatten()
    bg_values = bg_flat[np.where(bg_flat!=0)]
    bg_intensity = np.average(bg_values) #B

    edge_cont = edge_contrast(image_data,seg_data,k=k) #C

    ROI_size = np.sum(seg_data) / ROI #S

    SNR_diff = edge_cont * np.sqrt(ROI_size /bg_intensity)
    
    return SNR_diff


######### 10. Edge Contrast  #############

def edge_contrast(img_data,seg_data,ROI=1,k=1,verbose=False,plot=False):

    # Find edge voxels
    seg_data[seg_data==ROI] = 1
    seg_data[seg_data!=ROI] = 0
    inv_seg_data = 1 - seg_data
    edge_vox = seg_data - erode(seg_data,n=1)

    if verbose:
        print(np.sum(edge_vox),"edge voxels")

    # Find neighbouring voxels
    ROI_neighbours = convolve(seg_data,np.ones([2*k+1,2*k+1,2*k+1]),mode='constant',cval=0)
    non_ROI_neighbours = (2*k+1)**3 - ROI_neighbours

    # Validating that there is no mistake with the algorithm
    ROI_neighbours_edge = ROI_neighbours * edge_vox
    non_ROI_neighbours_edge = non_ROI_neighbours * edge_vox
    
    edge_max = max(np.amax(ROI_neighbours_edge),np.amax(non_ROI_neighbours_edge))
    assert edge_max < (2*k+1)**3, "Something has gone wrong in edge detection"


    # Find average intensities of surrounding image voxels and segmentation voxels for each voxel

    IMG_seg = seg_data*img_data
    IMG_inv = inv_seg_data*img_data

    IMG_seg_sum = convolve(IMG_seg,np.ones([2*k+1,2*k+1,2*k+1]),mode='constant',cval=0)
    IMG_inv_sum = convolve(IMG_inv,np.ones([2*k+1,2*k+1,2*k+1]),mode='constant',cval=0)

    ROI_neighbours[ROI_neighbours==0]=1
    non_ROI_neighbours[non_ROI_neighbours==0]=1

    
    IMG_seg_avg = np.nan_to_num(IMG_seg_sum / ROI_neighbours)*edge_vox
    IMG_inv_avg = np.nan_to_num(IMG_inv_sum / non_ROI_neighbours)*edge_vox

    if verbose:
        print(np.amax(IMG_seg_avg))
        print(np.amax(IMG_inv_avg))

    # Find local contrast for each voxel
    contrast_local = IMG_seg_avg - IMG_inv_avg
    contrasts = contrast_local[np.where(contrast_local!=0)]

    if plot:
        plt.hist(contrasts,bins=100)
        plt.show()

    # Find average edge contrast
    contrast_avg = np.sum(np.abs(contrast_local))/np.sum(edge_vox)

    return contrast_avg

def avg_to_edge(img_data,seg_data,k=1):

    return avg_intensity(img_data,seg_data)/edge_contrast(img_data,seg_data,k=k)


def gradient_contrast(image, segmentation,resolution):
    #Calculate gradient contrast at the edge of a segmentation in a 3D image.

    # Calculate the gradient of the image
    gradient = np.gradient(image,*resolution)
    gradient_magnitude = np.sqrt(gradient[0]**2+gradient[1]**2+gradient[2]**2)

    # Create a mask that selects the edges of the segmentation
    edge_mask = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            for k in range(segmentation.shape[2]):
                if segmentation[i,j,k]:
                    if i == 0 or i == segmentation.shape[0] - 1:
                        edge_mask[i,j,k] = 1
                    elif j == 0 or j == segmentation.shape[1] - 1:
                        edge_mask[i,j,k] = 1
                    elif k == 0 or k == segmentation.shape[2] - 1:
                        edge_mask[i,j,k] = 1
                    elif not segmentation[i-1,j,k] or not segmentation[i+1,j,k]:
                        edge_mask[i,j,k] = 1
                    elif not segmentation[i,j-1,k] or not segmentation[i,j+1,k]:
                        edge_mask[i,j,k] = 1
                    elif not segmentation[i,j,k-1] or not segmentation[i,j,k+1]:
                        edge_mask[i,j,k] = 1

    # Calculate the contrast at the edges of the segmentation
    edge_gradients = gradient_magnitude[edge_mask == 1]
    contrast = np.average(edge_gradients)

    return contrast

def grad_ratio(image, segmentation,resolution):

    # Calculate the gradient of the image
    gradient = np.gradient(image,*resolution)
    gradient_magnitude = np.sqrt(gradient[0]**2+gradient[1]**2+gradient[2]**2)
    avg_gradient = np.average(gradient_magnitude)

    # Create a mask that selects the edges of the segmentation
    edge_mask = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            for k in range(segmentation.shape[2]):
                if segmentation[i,j,k]:
                    if i == 0 or i == segmentation.shape[0] - 1:
                        edge_mask[i,j,k] = 1
                    elif j == 0 or j == segmentation.shape[1] - 1:
                        edge_mask[i,j,k] = 1
                    elif k == 0 or k == segmentation.shape[2] - 1:
                        edge_mask[i,j,k] = 1
                    elif not segmentation[i-1,j,k] or not segmentation[i+1,j,k]:
                        edge_mask[i,j,k] = 1
                    elif not segmentation[i,j-1,k] or not segmentation[i,j+1,k]:
                        edge_mask[i,j,k] = 1
                    elif not segmentation[i,j,k-1] or not segmentation[i,j,k+1]:
                        edge_mask[i,j,k] = 1

    # Calculate the contrast at the edges of the segmentation
    edge_gradients = gradient_magnitude[edge_mask == 1]
    contrast = np.average(edge_gradients)
    avg_grad_ratio = contrast / avg_gradient
    return avg_grad_ratio



########## 0. Setup & run ########


def image_physics(IMAGE_ID,plots=False,k=1):    

    image_file = (r"C:\Users\halja\Desktop\RESULTS\data\images\uncropped\NAXIVA_"+"%04d"+r"_0000.nii.gz") % IMAGE_ID
    gt_segmentation_file = (r"C:\Users\halja\Desktop\RESULTS\data\labels\uncropped\NAXIVA_"+"%04d"+r"_0000.nii.gz") % IMAGE_ID
    noise_segmentation_file = (r"C:\Users\halja\Desktop\MyProjectCode\pt-3-project\noise_segmentations\NAXIVA_"+"%04d"+r"_0000_Segmentation.nii") % IMAGE_ID

    image = nib.load(image_file)
    gt_seg = nib.load(gt_segmentation_file)
    noise_seg = nib.load(noise_segmentation_file)

    image_data = image.get_fdata()
    gt_seg_data = gt_seg.get_fdata()
    noise_seg_data = noise_seg.get_fdata()

    image_shape = image_data.shape

    res_img = resolution(image)
    print("Image resolution (mm):",res_img)
    res_seg = resolution(gt_seg)
    assert res_seg == res_img, "resolutions of image and segmentation not the same -- should really not be the case"

    vox_vol = np.prod(res_img)
    print("Voxel volume (mm^3):",vox_vol)

    avg_int = avg_intensity(image_data,gt_seg_data)
    print("Average intensity (ROI):",avg_int)

    standard_deviation = stdev(image_data,gt_seg_data)
    print("Standard deviation (ROI):",standard_deviation)

    standard_deviation_im = stdev(image_data,np.ones(image_data.shape))
    print("Standard deviation (Image):",standard_deviation_im)

    edge_cont = edge_contrast(image_data,gt_seg_data,k=k,plot=plots)
    print("Edge contrast:",edge_cont)

    gradient = gradient_contrast(image_data,gt_seg_data,res_img)
    print("Gradient:",gradient)

    avg_grad_ratio = grad_ratio(image_data,gt_seg_data,res_img)
    print("Avg grad ratio:",avg_grad_ratio)

    weber_cont = weber_contrast(image_data,gt_seg_data,k=3)
    print("Weber contrast:",weber_cont)

    ROI_size = np.sum(gt_seg_data)
    print("ROI size:",ROI_size)

    noise_seg_size = np.sum(noise_seg_data)
    print("Noise seg size:",noise_seg_size)

    img_noise = noise(noise_seg_data,image_data)
    print("Standard deviation of noise:",img_noise)

    if plots:
        noise_spectrum(noise_seg_data,image_data)
        print(" ")

    SNR = signal_noise_ratio(image_data,gt_seg_data,noise_seg_data)

    CNR = contrast_noise_ratio(image_data,gt_seg_data,noise_seg_data,k=k)

    ROSE = rose_model(image_data,gt_seg_data)
    print("Rose score:",ROSE)
    A2E = avg_int / edge_cont # Inverse of average edge contrast
    print("avg_to_edge",A2E)

    return res_img,vox_vol,avg_int,edge_cont,weber_cont,ROI_size,noise_seg_size,img_noise,SNR,CNR,ROSE,A2E,gradient,avg_grad_ratio,standard_deviation, standard_deviation_im





def correlate(feature_1,feature_2,RESULTS):
    f1 = [RESULTS[i][feature_1] for i in range(45)]
    f2 = [RESULTS[i][feature_2] for i in range(45)]

    f1 = f1 - np.mean(f1)
    f2 = f2 - np.mean(f2)

    f1_2 = np.sum(f1**2)
    f2_2 = np.sum(f2**2)

    corr = np.correlate(f1,f2) / np.sqrt(f1_2*f2_2)
    print("Correlation between",feature_1,"and",feature_2,"is",corr)
    return corr





if __name__ == "__main__":

    run_calculation = True

    RESULTS = {}
    run_name = "newstdev"
    k=2 

    if run_calculation:
        for i in range(47):
            print("NAXIVA_%04d" % i)

            IMAGE_ID = i

            RESULTS[i] = {}

            res,voxvol,avg,edge_cont,weber_cont,ROI_size,noise_size,img_noise,SNR,CNR,ROSE,A2E,gradient,avg_grad_ratio,standard_deviation,standard_deviation_im = image_physics(IMAGE_ID,k=k)

            RESULTS[i]['res'] = res
            RESULTS[i]['voxvol'] = voxvol
            RESULTS[i]['avg'] = avg
            RESULTS[i]['edge'] = edge_cont
            RESULTS[i]['weber'] = weber_cont
            RESULTS[i]['ROI_size'] = ROI_size
            RESULTS[i]['noise_size'] = noise_size
            RESULTS[i]['noise'] = img_noise
            RESULTS[i]['SNR'] = SNR
            RESULTS[i]['CNR'] = CNR
            RESULTS[i]['ROSE'] = ROSE
            RESULTS[i]['A2E'] = A2E
            RESULTS[i]['grad'] = gradient
            RESULTS[i]['grad_by_avggrad'] = avg_grad_ratio
            RESULTS[i]['STDEV'] = standard_deviation
            RESULTS[i]['STDEV'] = standard_deviation_im

            print(" ")
        
        with open(run_name+'.json', 'w') as fp:
            json.dump(RESULTS, fp)
    else:
        with open(run_name+'.json', 'r') as fp:
            RESULTS = json.load(fp)
        for i in range(45):
            RESULTS[i] = RESULTS[str(i)]
    #correlate("ROI_size","edge",RESULTS)
    #correlate("ROI_size","CNR",RESULTS)
    #correlate("ROI_size","SNR",RESULTS)
    #correlate("weber","edge",RESULTS)
    