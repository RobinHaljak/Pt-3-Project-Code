import json
import numpy as np
import nibabel as nib
from evaluation.physics import bbox2_3D
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path
from os import mkdir
from os.path import join


# 1. Find bounding box vertices + COM for each reference ROI -> 36 points
# 2. Check training set for which octiles the ROI falls into
# 3. Check for each point whether you can always exclude some octiles -- if these octiles are sufficient then can remove slices of the images according to them :)

def load_images(image_file,AMOS_file,GT_seg_file):

    img = nib.load(image_file)
    amos = nib.load(AMOS_file)
    gt = nib.load(GT_seg_file)

    img_data = img.get_fdata()
    amos_data = amos.get_fdata()
    gt_data = gt.get_fdata()
    
    hdr = img.header
    affine = img.affine

    assert img_data.shape == amos_data.shape == gt_data.shape, "All images should be cropped"
    print("Shape:",img_data.shape)
    
    return img_data, amos_data, gt_data, affine, hdr

def find_points(amos_data,verbose=True):
    points = []

    for i in range(1,5):
        seg_ROI = amos_data.copy()
        seg_ROI[seg_ROI!=i] = 0
        seg_ROI[seg_ROI==i] = 1

        # Find bounding box
        bounds = bbox2_3D(seg_ROI)
        
        if verbose:
            print("ROI",i,"BOUNDS:   X:",bounds[0],bounds[1],"  Y:",bounds[2],bounds[3],"  Z:",bounds[4],bounds[5])

        # Determine centre of mass
        COM = ndimage.center_of_mass(seg_ROI)
        
        if verbose:
            print("COM coordinates:","X:",COM[0],"Y:",COM[1],"Z:",COM[2])
        
        # Vertices of bounding box
        points.append([bounds[0],bounds[2],bounds[4]])
        points.append([bounds[0],bounds[2],bounds[5]])
        points.append([bounds[0],bounds[3],bounds[4]])
        points.append([bounds[0],bounds[3],bounds[5]])
        points.append([bounds[1],bounds[2],bounds[4]])
        points.append([bounds[1],bounds[2],bounds[5]])
        points.append([bounds[1],bounds[3],bounds[4]])
        points.append([bounds[1],bounds[3],bounds[5]])

        # Centre of Mass
        points.append([round(COM[0]),round(COM[1]),round(COM[2])])

    return points

def determine_octiles(gt_data,points,point_octiles):
    # Determine octile occupation for a single image given the segmentations and point locations
    for j in range(len(points)):
        if np.sum(gt_data[points[j][0]:,points[j][1]:,points[j][2]:]) > 0:
            point_octiles[j].append(1)
        if np.sum(gt_data[points[j][0]:,points[j][1]:,:points[j][2]]) > 0:
            point_octiles[j].append(2)
        if np.sum(gt_data[points[j][0]:,:points[j][1],points[j][2]:]) > 0:
            point_octiles[j].append(3)
        if np.sum(gt_data[points[j][0]:,:points[j][1],:points[j][2]]) > 0:
            point_octiles[j].append(4)
        if np.sum(gt_data[:points[j][0],points[j][1]:,points[j][2]:]) > 0:
            point_octiles[j].append(5)
        if np.sum(gt_data[:points[j][0],points[j][1]:,:points[j][2]]) > 0:
            point_octiles[j].append(6)
        if np.sum(gt_data[:points[j][0],:points[j][1],points[j][2]:]) > 0:
            point_octiles[j].append(7)
        if np.sum(gt_data[:points[j][0],:points[j][1],:points[j][2]]) > 0:
            point_octiles[j].append(8)

    return point_octiles

def remove_rare_octiles(point_octiles,set_octiles,small_count=0):
    # Do not count octile as occupied if it was occupied only small_count times

    for k in range(36):
        print("POINT",k)
        for j in range(1,9):
            print("Octile",j,"  count:",point_octiles[k].count(j))
        print("")

    for k in range(36):
        for j in range(1,9):
            if point_octiles[k].count(j) <= small_count:
                set_octiles[k].discard(j)

    return set_octiles

def create_crop_mask(points,img_data,set_octiles,boolean = True):

    # Create crop mask given set octiles

    crop_mask = np.zeros(img_data.shape)

    for j in range(36):
        if 1 not in set_octiles[j]:
            crop_mask[points[j][0]:,points[j][1]:,points[j][2]:] += 1
        if 2 not in set_octiles[j]:
            crop_mask[points[j][0]:,points[j][1]:,:points[j][2]] += 1
        if 3 not in set_octiles[j]:
            crop_mask[points[j][0]:,:points[j][1],points[j][2]:] += 1
        if 4 not in set_octiles[j]:
            crop_mask[points[j][0]:,:points[j][1],:points[j][2]] += 1
        if 5 not in set_octiles[j]:
            crop_mask[:points[j][0],points[j][1]:,points[j][2]:] += 1
        if 6 not in set_octiles[j]:
            crop_mask[:points[j][0],points[j][1]:,:points[j][2]] += 1
        if 7 not in set_octiles[j]:
            crop_mask[:points[j][0],:points[j][1],points[j][2]:] += 1
        if 8 not in set_octiles[j]:
            crop_mask[:points[j][0],:points[j][1],:points[j][2]] += 1

    if boolean:
        crop_mask[crop_mask==0] = -1
        crop_mask[crop_mask>0] = 0
        crop_mask[crop_mask==-1] = 1


    return crop_mask


def save_cropped(img_data,gt_data,bounds,crop_mask,fname_image,fname_label,affine,hdr,ID,mode='box'):

    # Save pre-cropped image

    #img_data is the image
    #gt_data is the segmentation

    if mode == 'box':
        img_data = img_data[bounds[ID][0]:bounds[ID][1],bounds[ID][2]:bounds[ID][3],bounds[ID][4]:bounds[ID][5]]
        gt_data = gt_data[bounds[ID][0]:bounds[ID][1],bounds[ID][2]:bounds[ID][3],bounds[ID][4]:bounds[ID][5]]
        
        img = nib.Nifti1Image(img_data, affine,hdr)
        nib.save(img, fname_image)

        seg = nib.Nifti1Image(gt_data, affine,hdr)
        nib.save(seg, fname_label)  

    elif mode == 'mask':
        img_data = img_data*crop_mask
        img_data = img_data[bounds[ID][0]:bounds[ID][1],bounds[ID][2]:bounds[ID][3],bounds[ID][4]:bounds[ID][5]]
        gt_data = gt_data*crop_mask
        gt_data = gt_data[bounds[ID][0]:bounds[ID][1],bounds[ID][2]:bounds[ID][3],bounds[ID][4]:bounds[ID][5]]

        img = nib.Nifti1Image(img_data, affine,hdr)
        nib.save(img, fname_image)

        seg = nib.Nifti1Image(gt_data, affine,hdr)
        nib.save(seg, fname_label)  
    else:
        print("invalid crop mode")








if __name__ == "__main__":

    outname = "gonnaplot"
    configuration = "chamos_on_naxiva_0.2_[1.0, 1.0, 1.0, 1.0]_2"
    small_count = 1
    save_files = False
    mode = 'box'

    outname = outname + configuration +"_"+str(small_count)
    image_base = r"C:\Users\halja\Desktop\RESULTS\segmentation results\images\NAXIVA_"+("%04d")+"_0000.nii.gz"
    AMOS_base = r"C:\Users\halja\Desktop\RESULTS\data\chamos\postprocessed\\"+configuration+r"\NAXIVA_"+("%04d")+"_0000.nii.gz"
    GT_seg_base = r"C:\Users\halja\Desktop\RESULTS\segmentation results\labels\NAXIVA_"+("%04d")+".nii.gz"

    output_folder = r"C:\Users\halja\Desktop\RESULTS\data\precropping_output"
    Path(join(output_folder,outname,'images')).mkdir(parents=True, exist_ok=True)
    Path(join(output_folder,outname,'labels')).mkdir(parents=True, exist_ok=True)
    output_base_images = join(output_folder,outname,'images',"NAXIVA_"+("%04d")+"_0000.nii.gz")
    output_base_labels = join(output_folder,outname,'labels',"NAXIVA_"+("%04d")+"_0000.nii.gz")

    output_folder_crop = r"C:\Users\halja\Desktop\RESULTS\segmentation results\cropmasks"

    point_octiles = {}
    for k in range(36):
        point_octiles[k] = []
    set_octiles = {}    
    bounds = {}

    #IDs of images in training ste
    training_IDs = [1,2,3]

    # Determine spatial relations
    for ID in range(47):
        if ID in training_IDs:
            print("ID",ID)
            image_file = image_base % ID
            AMOS_file = AMOS_base % ID
            GT_seg_file = GT_seg_base % ID

            img_data,amos_data,gt_data, affine, hdr = load_images(image_file,AMOS_file,GT_seg_file)

            points = find_points(amos_data)

            point_octiles = determine_octiles(gt_data,points,point_octiles)

    # Gather unique octiles
    for k in range(len(points)):
        set_octiles[k] = set(point_octiles[k])

    # Remove rare octiles
    set_octiles = remove_rare_octiles(point_octiles,set_octiles,small_count=small_count)

    vol_ratios = []
    seg_ratios = []

    # Creating crop masks
    for ID in range(47):
        print("ID",ID)
        image_file = image_base % ID
        AMOS_file = AMOS_base % ID
        GT_seg_file = GT_seg_base % ID
        output_file_image = output_base_images % ID
        output_file_label = output_base_labels % ID

        img_data,amos_data,gt_data,affine,hdr = load_images(image_file,AMOS_file,GT_seg_file)

        points = find_points(amos_data,verbose=False)
        crop_mask = create_crop_mask(points,img_data,set_octiles)

        bounds[ID] = bbox2_3D(crop_mask)
        bounds[ID] = tuple([int(i) for i in bounds[ID]])
        print(bounds[ID])
        
        bounded_crop_mask = np.zeros(crop_mask.shape)

        bounded_crop_mask[bounds[ID][0]:bounds[ID][1],bounds[ID][2]:bounds[ID][3],bounds[ID][4]:bounds[ID][5]] = 1
    
        crop_img = nib.Nifti1Image(crop_mask, affine,hdr)
        nib.save(crop_img, join(output_folder_crop,str(ID)+".nii.gz"))

        bounded_crop_img = nib.Nifti1Image(bounded_crop_mask, affine,hdr)
        nib.save(bounded_crop_img, join(output_folder_crop,str(ID)+"b.nii.gz"))

        

        # Determining cropping loss
        

        vol_ratios.append(((bounds[ID][1]-bounds[ID][0])*(bounds[ID][3]-bounds[ID][2])*(bounds[ID][5]-bounds[ID][4]))/(img_data.shape[0]*img_data.shape[1]*img_data.shape[2]))
        seg_ratios.append(gt_data[bounds[ID][0]:bounds[ID][1],bounds[ID][2]:bounds[ID][3],bounds[ID][4]:bounds[ID][5]].sum()/gt_data.sum())

        if save_files:
            print("Saving files:",output_file_image,output_file_label)
            save_cropped(img_data,gt_data,bounds,crop_mask,output_file_image,output_file_label,affine,hdr,mode=mode)
            

        if mode == 'box':
            print(gt_data.sum(),gt_data[bounds[ID][0]:bounds[ID][1],bounds[ID][2]:bounds[ID][3],bounds[ID][4]:bounds[ID][5]].sum())
            print("Volume ratio:",(bounds[ID][1]-bounds[ID][0])*(bounds[ID][3]-bounds[ID][2])*(bounds[ID][5]-bounds[ID][4]),img_data.shape[0]*img_data.shape[1]*img_data.shape[2],((bounds[ID][1]-bounds[ID][0])*(bounds[ID][3]-bounds[ID][2])*(bounds[ID][5]-bounds[ID][4]))/(img_data.shape[0]*img_data.shape[1]*img_data.shape[2]))
            print("Seg ratio:",gt_data[bounds[ID][0]:bounds[ID][1],bounds[ID][2]:bounds[ID][3],bounds[ID][4]:bounds[ID][5]].sum()/gt_data.sum())
        elif mode == 'mask':
            print("mask offf")
        else:
            print("Invalid Mode!")


    print(vol_ratios)
    print(seg_ratios)
    print(np.average(vol_ratios),np.amax(vol_ratios),np.amin(vol_ratios))
    print(np.average(seg_ratios),np.amax(seg_ratios),np.amin(seg_ratios))

    for k in range(len(points)):
        set_octiles[k] = list(point_octiles[k])

    output = [bounds,set_octiles,vol_ratios,seg_ratios,[np.average(vol_ratios),np.amax(vol_ratios),np.amin(vol_ratios)],[np.average(seg_ratios),np.amax(seg_ratios),np.amin(seg_ratios)]]

    # Save cropping boundary data
    with open(join(output_folder,outname,'bounds.json'), 'w') as fp:
        json.dump(output, fp)


    

