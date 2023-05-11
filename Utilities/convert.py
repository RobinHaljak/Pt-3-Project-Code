import sys
import os
from os import listdir
from os.path import join
import glob

import numpy as np
from skimage import draw

import csv
import nibabel as nib
import pydicom

### Convert code from .csv segmentations (Osirix) and DICOM images into nifti segmentations and images.

### Uses code written by David Gobbi (Adapted from https://gist.github.com/dgobbi/ab71f5128aa43f0d33a41775cb2bcca6)
### And code adapted from MATLAB code written by Mireia Crispin-Ortuzar.
### Code also used from https://github.com/scikit-image/scikit-image/issues/1103


def convert(path_in,path_out):
    # Adapted from https://gist.github.com/dgobbi/ab71f5128aa43f0d33a41775cb2bcca6
    sys.stdout.write("Converting DICOM to NIFTI.\n")
    sys.stdout.write("DICOM: %s\n" % path_in)
    sys.stdout.write("NIFTI: %s\n" % path_out)
    sys.stdout.flush()
    files = find_dicom_files(path_in)
    if not files:
        sys.stderr.write("No DICOM files found.\n")
        return 1
    series = load_dicom_series(files)
    if not series:
        sys.stderr.write("Unable to read DICOM files.\n")
        return 1
    vol, pixdim, mat = dicom_to_volume(series)
    convert_coords(vol, mat)
    write_nifti(path_out, vol, mat)

def find_dicom_files(path):
    # From https://gist.github.com/dgobbi/ab71f5128aa43f0d33a41775cb2bcca6
    """Search for DICOM files at the provided location.
    """
    if os.path.isdir(path):
        # check for common DICOM suffixes
        for ext in ("*.dcm", "*.DCM", "*.dc", "*.DC", "*.IMG"):
            pattern = os.path.join(path, ext)
            files = glob.glob(pattern)
            if files:
                break
        # if no files with DICOM suffix are found, get all files
        if not files:
            pattern = os.path.join(path, "*")
            contents = glob.glob(pattern)
            files = [f for f in contents if os.path.isfile(f)]
    else:
        sys.stderr.write("Cannot open %s\n" % (path,))
        return []

    return files

def write_nifti(filename, vol, affine):
    """Write a nifti file with an affine matrix.
    """
    output = nib.Nifti1Image(vol.T, affine)
    nib.save(output, filename)

def load_dicom_series(files):
    """Load a series of dicom files and return a list of datasets.

    The resulting list will be sorted by InstanceNumber.
    """
    # start by sorting filenames lexically
    sorted_files = sorted(files)

    # create list of tuples (InstanceNumber, DataSet)
    dataset_list = []
    for f in files:
        ds = pydicom.dcmread(f)
        try:
            i = int(ds.InstanceNumber)
        except (AttributeError, ValueError):
            i = -1
        dataset_list.append( (i, ds) )

    # sort by InstanceNumber (the first element of each tuple)
    dataset_list.sort(key=lambda t: t[0])

    # get the dataset from each tuple
    series = [t[1] for t in dataset_list]

    return series

def convert_coords(vol, mat):
    """Convert a volume from DICOM coords to NIFTI coords or vice-versa.

    For DICOM, x increases to the left and y increases to the back.
    For NIFTI, x increases to the right and y increases to the front.
    The conversion is done in-place (volume and matrix are modified).
    """
    # the x direction and y direction are flipped
    convmat = np.eye(4)
    convmat[0,0] = -1.0
    convmat[1,1] = -1.0

    # apply the coordinate change to the matrix
    mat[:] = np.dot(convmat, mat)

    # look for x and y elements with greatest magnitude
    xabs = np.abs(mat[:,0])
    yabs = np.abs(mat[:,1])
    xmaxi = np.argmax(xabs)
    yabs[xmaxi] = 0.0
    ymaxi = np.argmax(yabs)

    # re-order the data to ensure these elements aren't negative
    # (this may impact the way that the image is displayed, if the
    # software that displays the image ignores the matrix).
    if mat[xmaxi,0] < 0.0:
        # flip x
        vol[:] = np.flip(vol, 2)
        mat[:,3] += mat[:,0]*(vol.shape[2] - 1)
        mat[:,0] = -mat[:,0]
    if mat[ymaxi,1] < 0.0:
        # flip y
        vol[:] = np.flip(vol, 1)
        mat[:,3] += mat[:,1]*(vol.shape[1] - 1)
        mat[:,1] = -mat[:,1]

    # eliminate "-0.0" (negative zero) in the matrix
    mat[mat == 0.0] = 0.0

def dicom_to_volume(dicom_series):
    """Convert a DICOM series into a float32 volume with orientation.

    The input should be a list of 'dataset' objects from pydicom.
    The output is a tuple (voxel_array, voxel_spacing, affine_matrix)
    """
    # Create numpy arrays for volume, pixel spacing (ps),
    # slice position (ipp or ImagePositinPatient), and
    # slice orientation (iop or ImageOrientationPatient)
    n = len(dicom_series)
    shape = (n,) + dicom_series[0].pixel_array.shape
    vol = np.empty(shape, dtype=np.float32)
    ps = np.empty((n,2), dtype=np.float64)
    ipp = np.empty((n,3), dtype=np.float64)
    iop = np.empty((n,6), dtype=np.float64)

    for i, ds in enumerate(dicom_series):
        # create a single complex-valued image from real,imag
        image = ds.pixel_array
        try:
            slope = float(ds.RescaleSlope)
        except (AttributeError, ValueError):
            slope = 1.0
        try:
            intercept = float(ds.RescaleIntercept)
        except (AttributeError, ValueError):
            intercept = 0.0
        vol[i,:,:] = image*slope + intercept
        ps[i,:] = dicom_series[i].PixelSpacing
        ipp[i,:] = dicom_series[i].ImagePositionPatient
        iop[i,:] = dicom_series[i].ImageOrientationPatient

    # create nibabel-style affine matrix and pixdim
    # (these give DICOM LPS coords, not NIFTI RAS coords)
    affine, pixdim = create_affine(ipp, iop, ps)
    return vol, pixdim, affine

def create_affine(ipp, iop, ps):
    """Generate a NIFTI affine matrix from DICOM IPP and IOP attributes.

    The ipp (ImagePositionPatient) parameter should an Nx3 array, and
    the iop (ImageOrientationPatient) parameter should be Nx6, where
    N is the number of DICOM slices in the series.

    The return values are the NIFTI affine matrix and the NIFTI pixdim.
    Note the the output will use DICOM anatomical coordinates:
    x increases towards the left, y increases towards the back.
    """
    # solve Ax = b where x is slope, intecept
    n = ipp.shape[0]
    A = np.column_stack([np.arange(n), np.ones(n)])
    x, r, rank, s = np.linalg.lstsq(A, ipp, rcond=None)
    # round small values to zero
    x[(np.abs(x) < 1e-6)] = 0.0
    vec = x[0,:] # slope
    pos = x[1,:] # intercept

    # pixel spacing should be the same for all image
    spacing = np.ones(3)
    spacing[0:2] = ps[0,:]
    if np.sum(np.abs(ps - spacing[0:2])) > spacing[0]*1e-6:
        sys.stderr.write("Pixel spacing is inconsistent!\n");

    # compute slice spacing
    spacing[2] = np.round(np.sqrt(np.sum(np.square(vec))), 7)

    # get the orientation
    iop_average = np.mean(iop, axis=0)
    u = iop_average[0:3]
    u /= np.sqrt(np.sum(np.square(u)))
    v = iop_average[3:6]
    v /= np.sqrt(np.sum(np.square(v)))

    # round small values to zero
    u[(np.abs(u) < 1e-6)] = 0.0
    v[(np.abs(v) < 1e-6)] = 0.0

    # create the matrix
    mat = np.eye(4)
    mat[0:3,0] = u*spacing[0]
    mat[0:3,1] = v*spacing[1]
    mat[0:3,2] = vec
    mat[0:3,3] = pos

    # check whether slice vec is orthogonal to iop vectors
    dv = np.dot(vec, np.cross(u, v))
    qfac = np.sign(dv)
    if np.abs(qfac*dv - spacing[2]) > 1e-6:
        sys.stderr.write("Non-orthogonal volume!\n");

    # compute the nifti pixdim array
    pixdim = np.hstack([np.array(qfac), spacing])

    return mat, pixdim

def convert_coords(vol, mat):
    """Convert a volume from DICOM coords to NIFTI coords or vice-versa.

    For DICOM, x increases to the left and y increases to the back.
    For NIFTI, x increases to the right and y increases to the front.
    The conversion is done in-place (volume and matrix are modified).
    """
    # the x direction and y direction are flipped
    convmat = np.eye(4)
    convmat[0,0] = -1.0
    convmat[1,1] = -1.0

    # apply the coordinate change to the matrix
    mat[:] = np.dot(convmat, mat)

    # look for x and y elements with greatest magnitude
    xabs = np.abs(mat[:,0])
    yabs = np.abs(mat[:,1])
    xmaxi = np.argmax(xabs)
    yabs[xmaxi] = 0.0
    ymaxi = np.argmax(yabs)

    # re-order the data to ensure these elements aren't negative
    # (this may impact the way that the image is displayed, if the
    # software that displays the image ignores the matrix).
    if mat[xmaxi,0] < 0.0:
        # flip x
        vol[:] = np.flip(vol, 2)
        mat[:,3] += mat[:,0]*(vol.shape[2] - 1)
        mat[:,0] = -mat[:,0]
    if mat[ymaxi,1] < 0.0:
        # flip y
        vol[:] = np.flip(vol, 1)
        mat[:,3] += mat[:,1]*(vol.shape[1] - 1)
        mat[:,1] = -mat[:,1]

    # eliminate "-0.0" (negative zero) in the matrix
    mat[mat == 0.0] = 0.0

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    # From https://github.com/scikit-image/scikit-image/issues/1103
    # Draw a polymask around the segmentation vertices
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def DICOM_to_nifti(base_folder,output_folder,addition=""):
    # Travels the usual DICOM folder setup to find each DICOM folder
    # Specific to my used data setup - adapt to your own.
    patients = sorted(listdir(base_folder))
    dates = {}
    for pat in patients:
        patient_path = join(base_folder,pat)
        dates[pat] = sorted(listdir(patient_path))
        if '.DS_Store' in dates[pat]: 
            dates[pat].remove('.DS_Store')
        print("Patient: ",pat," Dates: ", dates[pat])

        i = 1
        for date in dates[pat]:
            
            one_more_folder = join(patient_path,date)
            next_folder = sorted(listdir(one_more_folder))
            if '.DS_Store' in next_folder: 
                next_folder.remove('.DS_Store')
            if len(next_folder) > 1:
                print("Something suspicious happened (more than 1 DICOM folder)")
            input_folder = join(one_more_folder,next_folder[0])

            output_name = pat+"_"+date+"_"+addition+".nii.gz"
            output_path = join(output_folder,output_name)

            convert(input_folder,output_path)

            i += 1

def create_ROI_mask(data,first_slide,image_size,desired_ROIs,ROI_indices):
    # Adapted from Mireia's MATLAB code
     
    # Get all unique ROIs
    roi_list = []
    for i in range(1,len(data)):
        roi_list.append(data[i][8])

    roi_mask_total = np.zeros(image_size)
    
    for k,ROI in enumerate(desired_ROIs):
        if ROI not in roi_list:
            # ROI missing (normal) or Invalid ROI name (not normal)
            continue
    
        N_slices = len(data)
        
        
        relevant_lines = []
        for slice in range(1,N_slices):
            if data[slice][8] == ROI:
                relevant_lines.append(slice)

        roi_mask = np.zeros(image_size,dtype=bool)

        # Turn data into contour
        mm_x = []
        mm_y = []
        mm_z = []
        cor_px_y = []
        cor_px_x = []
        cor_px_z = []

        for icount in range(len(relevant_lines)):
            i = relevant_lines[icount]
            # Vector of mm

            points = np.array([float(x) for x in data[i][25:] if x != '']) 
            #'' occurs as an empty value in the import data sometimes, if there are any other values you wish to eclude then just add them to the list comprehension

            filled_points = points[~np.isnan(points)]
            mm_x += filled_points[0::5].tolist()
            mm_y += filled_points[1::5].tolist()
            mm_z += filled_points[2::5].tolist()

            # Vector of pixels
            slice_num = int(data[i][0]) - first_slide + 1 #NB! +1 necessary since ImageNo gets counted from 0, but the image names get counted from 1 :)
            num_points = int(data[i][24])
            slice_reps = np.ones(num_points, dtype='int') * slice_num
            cor_px_x += filled_points[3::5].tolist()
            cor_px_y += slice_reps.tolist()
            cor_px_z += filled_points[4::5].tolist()

            # Create Mask
            border_x = filled_points[3::5]
            border_z = filled_points[4::5]
            centerx = border_x[0]
            centerz = border_z[0]
            border_x = np.append(border_x, centerx)
            border_z = np.append(border_z, centerz)


            if len(border_x) <= 2:
                print("NOT ENOUGH VERTICES TO CREATE MASK") #Need at least 3 vertices to draw a mask


            xy_slice_filled = poly2mask(border_x,border_z,(image_size[0], image_size[1]))
            roi_mask[:, :, slice_num] = np.bitwise_or(roi_mask[:, :, slice_num], xy_slice_filled)

        ### NB! Closeby ROIs may overlap, in which case the last ROI in desired_ROIs will remain if the commented out line is used
        ### This may occur even if the original polygon masks (defined by the vertices) do not overlap -- the chance for this is very small though
        ### nifti segmentation files don't support overlapping segmentations (afaik) so until the ROIs are not overlapping to begin with then there is no worry
        ### The lower line will create a new ROI with a new index (which could overlap with an existing index...) if the segmentations overlap - but is overall safer
        ### It is not really relevant for me to properly fix this issue (nor is it really possible), so may this just serve as a warning

        #roi_mask_total[np.where(roi_mask!=0)] = ROI_indices[k]
        roi_mask_total += ROI_indices[k] * roi_mask

    roi_mask_total = np.flip(roi_mask_total, axis=0)
    roi_mask_total = np.flip(roi_mask_total, axis=1)

    return roi_mask_total

def import_data(image_file,csv_file):
    # Import image nifti (converted from dicom)
    image = nib.load(image_file)
    image_data = image.get_fdata()
    image_affine = image.affine
    image_hdr = image.header
    image_size = image_data.shape

    # Open the segmentation CSV file
    with open(csv_file, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)

        # Create an empty list to store the rows of data
        seg_data = []
            
        # Iterate over each row in the CSV file
        for row in reader:

            # Append the row to the list of data
            seg_data.append(row)

    return seg_data,image_data,image_affine,image_hdr,image_size



if __name__ == '__main__':

    ### MODIFY THESE:
    # Output folder for nifti images
    image_output_folder =r"C:\Users\halja\Desktop\RESULTS\new convert\images"
    # Output folder for nifti segmentations
    seg_output_folder = r"C:\Users\halja\Desktop\RESULTS\new convert\segmentations"
    # Base folder for DICOMS (only contains folders like NAXIVA_NXXXX)
    image_folder = r"C:\Users\halja\Desktop\RESULTS\new convert\dicoms"
    # Folder containing segmentation csvs
    csv_folder = r"C:\Users\halja\Desktop\RESULTS\new convert\csvs"

    desired_ROIs = ['Left VTT','Right VTT']
    ROI_indices = [1,1] # order corresponds to desired_ROIs



    ### Import segmentation data
    csv_files = sorted(listdir(csv_folder))

    

    ### Find index of first slide in DICOM folder
    first_slide = {}
    for folder0 in sorted(listdir(image_folder)):
        if folder0 != '.DS_Store':
            for folder1 in sorted(listdir(join(image_folder,folder0))):
                if folder1 != '.DS_Store':
                    for folder2 in sorted(listdir(join(image_folder,folder0,folder1))):
                        if folder2 != '.DS_Store':
                            DICOM_folder = join(image_folder,folder0,folder1,folder2)
                            dicom_names = sorted(listdir(DICOM_folder))
                            if '.DS_Store' in dicom_names: 
                                dicom_names.remove('.DS_Store')
                            first_slide[folder0+"_"+folder1] = int(dicom_names[0][-8:-4])
    
    scan_identifiers = sorted(first_slide.keys())

    ### Convert DICOMs to nifti
    #Can comment the below line out if the DICOM to nifti conversion has already been run once on this image data
    
    DICOM_to_nifti(image_folder,image_output_folder,addition="") #addition is a string that gets added to the end of the default image name (e.g. "A" for axial images)

    ### Import scan data

    nifti_files = sorted(listdir(image_output_folder))
    print(csv_files,nifti_files)
    print(len(nifti_files), "image files and",len(csv_files),"segmentation files")

    # Check that image and segmentation names matching

    for n,csv_file in enumerate(csv_files):
        if csv_file[:21].replace(" ","_") != nifti_files[n][:21]:
            print("WARNING:",csv_file[:21].replace(" ","_"),nifti_files[n][:21])

    # Create segmentation nifti

    for n,seg_file in enumerate(csv_files):

          
        image_path = join(image_output_folder,nifti_files[n])
        seg_path= join(csv_folder,seg_file)

        scan_identifier = scan_identifiers[n]
        print("Creating:",scan_identifier) 

        # Import data
        seg_data,image_data,image_affine,image_hdr,image_size = import_data(image_path,seg_path)

        # Create segmentation mask
        roi_mask = create_ROI_mask(seg_data,first_slide[scan_identifier],image_size,desired_ROIs,ROI_indices)

        # Save nifti segmentation
        mask = nib.Nifti1Image(roi_mask, image_affine,image_hdr)
        nib.save(mask, join(seg_output_folder,nifti_files[n]))  

