import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from skimage import morphology, exposure
import scipy
from scipy import ndimage
import h5py
import torch

v = "v1"

class CFG:
    do_windowing = True
    window_width = 2000  # -160
    window_center = 350  # 240

    do_background_cropping = True
    cropping_width = 0.45
    cropping_center = 0.5

    do_cropping = True
    do_mask_cropping = True

    do_spacing = False
    target_spacing = [1, 1, 1]

    do_reshape = True
    new_size = [192, 192, 96]       # v1


# windowing
def transform_ctdata(image, windowWidth, windowCenter, normal=False):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


# background removing
def image_background_segmentation(img, WW=40, WL=80):
    # print(img.shape)  # 512*512
    # Calculate the outside values by hand (again)
    lB = WW - WL
    uB = WW + WL

    # Keep only values inside of the window
    background_seperation = np.logical_and(img > lB, img < uB)      # (-40, 120)
    # print(background_seperation.shape)  # 512*512
    background_seperation = morphology.dilation(background_seperation, np.ones((5, 5)))
    labels, label_nb = scipy.ndimage.label(background_seperation)
    label_count = np.bincount(labels.ravel().astype(np.int))

    # discard the 0 label
    label_count[0] = 0
    mask = labels == label_count.argmax()  # find the most frequency number mask

    mask = morphology.dilation(mask, np.ones((4, 4)))  # dilate the mask for less fuzzy edges
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))  # dilate the mask again

    return mask, mask * img


# cropping
def crop(mask, vol):
    for i in range(mask.shape[0]):      # 125*512*512
        coords = np.array(np.nonzero(mask[i]))
        if i == 0:
            top_left = np.min(coords, axis = 1)
            bottom_right = np.max(coords, axis = 1)
        else:
            top_left = np.vstack((top_left, np.min(coords, axis = 1)))
            bottom_right = np.vstack((bottom_right, np.max(coords, axis = 1)))
    top = max(0, min(top_left[:, 0]) - 20)
    left = max(0, min(top_left[:, 1]) - 20)
    bottom = min(mask.shape[1], max(bottom_right[:, 0]) + 20)
    right = min(mask.shape[2], max(bottom_right[:, 1]) + 20)
    croped_vol = vol[:, top : bottom, left : right]
    return croped_vol


def getRangImageDepth(image):
    """
    :param image:
    :return:rangofimage depth
    """
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition


def make_patch(image, mask, startpostion, endpostion):
    """
    make number patch
    :param image:[depth,512,512]
    :return:[n,512,512]
    """
    imagezsrc = np.shape(image)[0]
    subimage_startpostion = startpostion - 10
    subimage_endpostion = endpostion + 10
    if subimage_startpostion < 0:
        subimage_startpostion = 0
    if subimage_endpostion > imagezsrc:
        subimage_endpostion = imagezsrc
    imageroi = image[subimage_startpostion:subimage_endpostion, :, :]
    maskroi = mask[subimage_startpostion:subimage_endpostion, :, :]
    return imageroi, maskroi

# reshape
def resampling(roiImg, new_size, lbl=False):
    new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                   zip(roiImg.GetSize(), roiImg.GetSpacing(), new_size)]
    if lbl:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    else:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkLinear, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())

    return resampled_sitk


# space resampling
def space_resampling(roiImg, new_spacing, lbl=False):
    # print('old spacing: ', roiImg.GetSpacing())
    new_size = [int(old_sz * old_spc / new_spc) for old_sz, old_spc, new_spc in
                zip(roiImg.GetSize(), roiImg.GetSpacing(), new_spacing)]

    if lbl:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    else:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkLinear, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())

    return resampled_sitk



loadPath = "../data/KiTS19/data/"
savePath = "../data/KiTS19/processed_" + v + "_h5/"
savePath2 = "../data/KiTS19/processed_" + v + "_nii/"
if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath2):
    os.makedirs(savePath2)

if __name__ == "__main__":
    basedir='../../data/MMWHS-CT'
    for i in range(1,21):
        path=os.path.join(basedir,'ct_train_10'+str(i).zfill(2))
        ct=sitk.ReadImage(path+'_image.nii.gz')
        lbl=sitk.ReadImage(path+'_label.nii.gz')
        print(ct)
        ct_array=sitk.GetArrayFromImage(ct)
        lbl_array=sitk.GetArrayFromImage(lbl)
        print(ct_array.shape)
        if CFG.do_windowing:
            ct_array = transform_ctdata(ct_array, CFG.window_width, CFG.window_center, True)
        if CFG.do_background_cropping:
            mask = np.zeros(ct_array.shape)
            for j in range(ct_array.shape[0]):
                mask[j], ct_array[j] = image_background_segmentation(ct_array[j], WW=CFG.cropping_center, WL=CFG.cropping_width)
        if CFG.do_cropping:
            ct_array = crop(mask, ct_array)
            lbl_array = crop(mask, lbl_array)
            print('crop shape:', ct_array.shape)
            print('crop shape:', lbl_array.shape)
        if CFG.do_mask_cropping:
            startpostion, endpostion = getRangImageDepth(lbl_array)
            ct_array, lbl_array = make_patch(ct_array, lbl_array, startpostion=startpostion, endpostion=endpostion)
            print('crop mask shape:', ct_array.shape)
            print('crop mask shape:', lbl_array.shape)




        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())
        new_ct.SetDirection(ct.GetDirection())
        new_lbl = sitk.GetImageFromArray(lbl_array)
        new_lbl.SetOrigin(lbl.GetOrigin())
        new_lbl.SetSpacing(lbl.GetSpacing())
        new_lbl.SetDirection(lbl.GetDirection())
        print(new_ct)
        if CFG.do_spacing:
            new_ct = space_resampling(new_ct, CFG.target_spacing, lbl=False)
            new_lbl = space_resampling(new_lbl, CFG.target_spacing, lbl=True)
        elif CFG.do_reshape:
            new_ct = resampling(new_ct, CFG.new_size, lbl=False)
            new_lbl = resampling(new_lbl, CFG.new_size, lbl=True)
        save_ct_array = sitk.GetArrayFromImage(new_ct)
        save_lbl_array = sitk.GetArrayFromImage(new_lbl)
        from_labels = [500, 600, 420, 550, 205, 820, 850]
        to_labels = [1, 2, 3, 4, 5, 6, 7]
        for from_label, to_label in zip(from_labels, to_labels):
            save_lbl_array[save_lbl_array == from_label] = to_label
        # set everything else to zero
        save_lbl_array[save_lbl_array > 7] = 0
        save_ct_array = save_ct_array.swapaxes(0, 2)
        save_lbl_array = save_lbl_array.swapaxes(0, 2)
        save_file = h5py.File('../../data/MMWHS/volume-'+str(i).zfill(2)+ ".h5", 'w')
        save_file.create_dataset('image', data=save_ct_array)
        save_file.create_dataset('label', data=save_lbl_array)
        print('array shape: ', save_ct_array.shape)
        save_file.close()
        sitk.WriteImage(new_ct,'../../data/MMWHS/volume-'+str(i).zfill(2)+'.nii.gz')
        sitk.WriteImage(new_lbl,'../../data/MMWHS/label-'+str(i).zfill(2)+'.nii.gz')
    exit(0)
    fileList = os.listdir(loadPath)
    for i in range(210):
        subpath = "case_" + str(i).zfill(5)
        print(subpath)
        ct = sitk.ReadImage(loadPath + subpath + "/imaging.nii.gz")
        lbl = sitk.ReadImage(loadPath + subpath + "/segmentation.nii.gz")
        ct_array = sitk.GetArrayFromImage(ct)
        lbl_array = sitk.GetArrayFromImage(lbl)
        ct_array = ct_array.swapaxes(0, 2)
        lbl_array = lbl_array.swapaxes(0, 2)
        new_ct_array = ct_array
        new_lbl_array = lbl_array
        print('raw img shape:', ct_array.shape)         # (611, 512, 512)

        if CFG.do_windowing:
            new_ct_array = transform_ctdata(new_ct_array, CFG.window_width, CFG.window_center, True)

        if CFG.do_background_cropping:
            mask = np.zeros(new_ct_array.shape)
            for i in range(new_ct_array.shape[0]):
                mask[i], new_ct_array[i] = image_background_segmentation(new_ct_array[i], WW=CFG.cropping_center, WL=CFG.cropping_width)

        if CFG.do_cropping:
            new_ct_array = crop(mask, new_ct_array)
            new_lbl_array = crop(mask, new_lbl_array)
            print('crop shape:', new_ct_array.shape)
            print('crop shape:', new_lbl_array.shape)

        if CFG.do_mask_cropping:
            startpostion, endpostion = getRangImageDepth(new_lbl_array)
            new_ct_array, new_lbl_array = make_patch(new_ct_array, new_lbl_array, startpostion=startpostion, endpostion=endpostion)
            print('crop mask shape:', new_ct_array.shape)
            print('crop mask shape:', new_lbl_array.shape)

        new_ct = sitk.GetImageFromArray(new_ct_array)
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())
        new_ct.SetDirection(ct.GetDirection())
        new_lbl = sitk.GetImageFromArray(new_lbl_array)
        new_lbl.SetOrigin(lbl.GetOrigin())
        new_lbl.SetSpacing(lbl.GetSpacing())
        new_lbl.SetDirection(lbl.GetDirection())

        if CFG.do_spacing:
            new_ct = space_resampling(new_ct, CFG.target_spacing, lbl=False)
            new_lbl = space_resampling(new_lbl, CFG.target_spacing, lbl=True)
        elif CFG.do_reshape:
            new_ct = resampling(new_ct, CFG.new_size, lbl=False)
            new_lbl = resampling(new_lbl, CFG.new_size, lbl=True)
        save_ct_array = sitk.GetArrayFromImage(new_ct)
        save_lbl_array = sitk.GetArrayFromImage(new_lbl)

        # output shape
        # print('new img shape:', save_ct_array.shape)
        # print('new lbl shape:', save_lbl_array.shape)

        # save
        # sitk.WriteImage(new_ct, os.path.join(savePath2, subpath + '_img.nii'))
        # sitk.WriteImage(new_lbl, os.path.join(savePath2, subpath + '_lbl.nii'))
        save_file = h5py.File(savePath + subpath + ".h5", 'w')
        save_file.create_dataset('image', data=save_ct_array)
        save_file.create_dataset('label', data=save_lbl_array)
        save_file.close()

