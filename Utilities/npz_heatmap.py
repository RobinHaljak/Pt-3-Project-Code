import cv2
import numpy as np
import nibabel as nib
import time
from segmentation_and_postprocessing.postprocessing import Left_Right_softmax_split

# Interactive viewing of softmax output of nnUNet.


def import_images(im_path,heatmap_path,ROI_index,LR=False):
    # Import sogtmax values and image values

    ## Load cropped image .npz
    if im_path[-4:] == ".npz":
        data = np.load(im_path)
        data = data["data"]

        image = data[0,:, :, :]
    elif im_path[-7:]==".nii.gz":
        img = nib.load(im_path)
        data = img.get_fdata()

        image = data
    else:
        print("File format of:",im_path,"not accepted")
        raise Exception("Bad image format")

    img_shape = image.shape
    print(img_shape)

     ## Load heatmap .npz
    data = np.load(heatmap_path)
    heatmap = data["softmax"]
    if LR:  
        heatmap = Left_Right_softmax_split(heatmap)
    print(heatmap.shape)
    #image = np.swapaxes(image,0,2)
    heatmap_shape = heatmap[ROI_index,:, :, :].shape
    assert img_shape == heatmap_shape,"Expect image and heatmap to be same shape - might need to pull the cropped image .npz from nnunnet cropped or preprocessed"

    return image, heatmap, img_shape

def view_heatmap(image,heatmap,slice_num,mask_on,ROI_index):
    print("Slice:",slice_num)
    img_shape = image.shape
    image3 = np.zeros([img_shape[1],img_shape[2],3])
    image3[:,:,0] = image[slice_num, :,:]
    image3[:,:,1] = image[slice_num, :,:]
    image3[:,:,2] = image[slice_num, :,:]
    max_val = np.amax(image)

    # Background image
    image3 = image3 / max_val # Normalising
    image3 = (255*image3).astype(np.uint8)

    # Softmax heatmap
    heatmap3 = (255*heatmap[ROI_index,slice_num, :, :]).astype(np.uint8)
    heatmap3 = cv2.applyColorMap(heatmap3, cv2.COLORMAP_JET)

    if mask_on:
        super_imposed_img = cv2.addWeighted(heatmap3, 0.4, image3, 0.6, 0)
        super_imposed_img = np.flip(super_imposed_img,(0,1))

        return super_imposed_img
    else:

        image3 = np.flip(image3,(0,1))

        return image3



def scroll_heatmap(im_path,npz_path):
    # Controls for viewing heatmap

    print("q - terminate, w - faster, s - slower, r - reverse, p - pause, u - unpause, a/d - one back/forward, e - mask on/off, z/c - ROI back/forward")
    slice_num = 0
    ROI_index = 1
    image3, heatmap3, image_shape = import_images(im_path,npz_path,ROI_index)

    num_z = image_shape[0]
    sleep_time = 0.1
    step = 1
    pause_on_next = False
    mask_on = True
    interpolation = True
    interpolation_size = ((int(image_shape[1]*2.5)),int(image_shape[2]*2.5))


    while True:

    # ROI / slice q
        slice_num += step

        slice_num = slice_num % num_z
        super_imposed_img = view_heatmap(image3,heatmap3,slice_num,mask_on,ROI_index)
        if interpolation:
            super_imposed_img = cv2.resize( super_imposed_img, interpolation_size, interpolation = cv2.INTER_CUBIC )

        cv2.imshow("image", super_imposed_img)
        time.sleep(sleep_time)
        
        ### VIEWER CONTROLS

        if cv2.waitKey(1) == ord('q'):
            
            # press q to terminate the loop
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) == ord('w'):
            
            sleep_time /= 2
            print("GOTTA GO FAST")
        if cv2.waitKey(1) == ord('s'):
            
            sleep_time *= 2
            print("SLOW DOWN MY GUY")
        if cv2.waitKey(1) == ord('p'):
            sleep_time = 1000
            print("PAUSE")
            while True:
                if cv2.waitKey(1) == ord('u'):
                    sleep_time = 0.1
                    print("UNPAUSE")
                    break
               
                if cv2.waitKey(1) == ord('a'):
                    
                    pause_on_next = True
                    sleep_time = 0.1
                    break
                if cv2.waitKey(1) == ord('d'):

                    pause_on_next = True
                    sleep_time = 0.1
                    break

                if cv2.waitKey(1) == ord('e'):
                    mask_on = not mask_on
                    
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0

                    print("Mask on/off")
                    break
                if cv2.waitKey(1) == ord('z'):
            
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0
                    ROI_index = (ROI_index-1)%5
                    print("ROI:",ROI_index)

                    break
                if cv2.waitKey(1) == ord('c'):
            
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0
                    ROI_index = (ROI_index+1)%5
                    print("ROI:",ROI_index)

                    break

        if cv2.waitKey(1) == ord('r'):
            step *= -1
            time.sleep(1)
            print("REVERSE")

        if cv2.waitKey(1) == ord('e'):
            mask_on = not mask_on

            print("Mask on/off")
        
        if pause_on_next:
            pause_on_next = False
            while True:
                if cv2.waitKey(1) == ord('u'):
                    sleep_time = 0.1
                    print("UNPAUSE")
                    break
                if cv2.waitKey(1) == ord('a'):
                    step = -1
                    pause_on_next = True
                    sleep_time = 0.1
                    break
                if cv2.waitKey(1) == ord('d'):
                    step = +1
                    pause_on_next = True
                    sleep_time = 0.1
                    break
                if cv2.waitKey(1) == ord('e'):
                    mask_on = not mask_on
                    
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0

                    print("Mask on/off")
                    break
                if cv2.waitKey(1) == ord('z'):
            
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0
                    ROI_index = (ROI_index-1)%5
                    print("ROI:",ROI_index)

                    break
                if cv2.waitKey(1) == ord('c'):
            
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0
                    ROI_index = (ROI_index+1)%5
                    print("ROI:",ROI_index)

                    break

















if __name__ == "__main__":

    #cv2.imwrite('color_img.jpg', img)

    ROI_index = 1
    slice_num = 0
    do_left_right_kidney_split = False
    im_path = "C:\\Users\\halja\\Desktop\\MyProjectCode\\Task506_NAXIVA\\NAXIVA_0003.npz"
    npz_path = r"C:\Users\halja\Desktop\MyProjectCode\inference_results\model_ep_060\nnUNetTrainerV2__nnUNetPlansv2.1\NAXIVA_0003_0000.npz"
    print("q - terminate, w - faster, s - slower, r - reverse, p - pause, u - unpause, a/d - one back/forward, e - mask on/off, z/c - ROI back/forward")

    image3, heatmap3, image_shape = import_images(im_path,npz_path,ROI_index,LR=do_left_right_kidney_split)

    num_z = image_shape[0]
    sleep_time = 0.1
    step = 1
    pause_on_next = False
    mask_on = True
    interpolation = True
    interpolation_size = ((int(image_shape[1]*2)),int(image_shape[2]*2))

    while True:

        slice_num += step

        slice_num = slice_num % num_z
        super_imposed_img = view_heatmap(image3,heatmap3,slice_num,mask_on,ROI_index) 
        if interpolation:
            super_imposed_img = cv2.resize( super_imposed_img, interpolation_size, interpolation = cv2.INTER_CUBIC ) #Interpolate image for easier viewing: note will mess with pixel by pixel views of segmentations

        cv2.imshow("image", super_imposed_img) #Open heatmap image

        time.sleep(sleep_time)

        if cv2.waitKey(1) == ord('q'):
            
            # press q to terminate the loop
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) == ord('w'):
            
            sleep_time /= 2
            print("GOTTA GO FAST")
        if cv2.waitKey(1) == ord('s'):
            
            sleep_time *= 2
            print("SLOW DOWN MY GUY")
        if cv2.waitKey(1) == ord('p'):
            sleep_time = 1000
            print("PAUSE")
            while True:
                if cv2.waitKey(1) == ord('q'):
            
                    # press q to terminate the loop
                    cv2.destroyAllWindows()
                    break
                if cv2.waitKey(1) == ord('u'):
                    sleep_time = 0.1
                    print("UNPAUSE")
                    break
                    
                if cv2.waitKey(1) == ord('a'):
                    
                    pause_on_next = True
                    sleep_time = 0.1
                    break
                if cv2.waitKey(1) == ord('d'):

                    pause_on_next = True
                    sleep_time = 0.1
                    break

                if cv2.waitKey(1) == ord('e'):
                    mask_on = not mask_on
                    
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0

                    print("Mask on/off")
                    break
                if cv2.waitKey(1) == ord('z'):
            
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0
                    ROI_index = (ROI_index-1)%5
                    print("ROI:",ROI_index)

                    break
                if cv2.waitKey(1) == ord('c'):
            
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0
                    ROI_index = (ROI_index+1)%5
                    print("ROI:",ROI_index)

                    break

        if cv2.waitKey(1) == ord('r'):
            step *= -1
            time.sleep(1)
            print("REVERSE")

        if cv2.waitKey(1) == ord('e'):
            mask_on = not mask_on

            print("Mask on/off")
        
        if pause_on_next:
            pause_on_next = False
            while True:
                if cv2.waitKey(1) == ord('q'):
            
                    # press q to terminate the loop
                    cv2.destroyAllWindows()
                    break
                if cv2.waitKey(1) == ord('u'):
                    sleep_time = 0.1
                    print("UNPAUSE")
                    break
                if cv2.waitKey(1) == ord('a'):
                    step = -1
                    pause_on_next = True
                    sleep_time = 0.1
                    break
                if cv2.waitKey(1) == ord('d'):
                    step = +1
                    pause_on_next = True
                    sleep_time = 0.1
                    break
                if cv2.waitKey(1) == ord('e'):
                    mask_on = not mask_on
                    
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0

                    print("Mask on/off")
                    break
                if cv2.waitKey(1) == ord('z'):
            
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0
                    ROI_index = (ROI_index-1)%5
                    print("ROI:",ROI_index)

                    break
                if cv2.waitKey(1) == ord('c'):
            
                    step = 0
                    pause_on_next = True
                    sleep_time = 1.0
                    ROI_index = (ROI_index+1)%5
                    print("ROI:",ROI_index)

                    break



        

