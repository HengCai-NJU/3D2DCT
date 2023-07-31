import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import ramps, losses


def test_all_case(net1, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None,dataset='la'):
    total_metric = 0.0
    totaldice=0.0
    totalasd=0.0
    totalhd=0.0
    totaliou=0.0
    dicelist=[]
    # delete after
    #slice_random=[5,0,3,11,3,7,9,3 , 5,  2 , 4,  7,  6,  8,  8, 10]
    #img_idx=0
    for image_path in tqdm(image_list):
        print(image_path)
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        if dataset=='mmwhs':
            image = (image - np.mean(image)) / np.std(image)


        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net1, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        classdice=[]
        classiou=[]
        classasd=[]
        classhd=[]
        for c in range(1,num_classes):
            if np.count_nonzero(prediction==c)==0 and np.count_nonzero(label[:]==c)!=0:
                curdice,curiou,curhd,curasd=(0,0,50,50)
            elif np.count_nonzero(prediction==c)==0 and np.count_nonzero(label[:]==c)==0:
                curdice,curiou,curhd,curasd=(1,1,0,0)
            else:
                curdice,curiou,curhd,curasd=calculate_metric_percase(prediction==c,label[:]==c)
            classdice.append(curdice)
            classiou.append(curiou)
            classhd.append(curhd)
            classasd.append(curasd)
        classdice.append(np.mean(classdice))
        classiou.append(np.mean(classiou))
        classhd.append(np.mean(classhd))
        classasd.append(np.mean(classasd))
        totaldice+=np.asarray(classdice)
        totalhd+=np.asarray(classhd)
        totaliou+=np.asarray(classiou)
        totalasd+=np.asarray(classasd)
        print(classdice)
        print(classiou)
        print(classhd)
        print(classasd)
        dicelist.append(classdice)

        if save_result:
            nib.save(nib.Nifti1Image(score_map.astype(np.float32), np.eye(4)), test_save_path + id + "_prob.nii.gz")
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            prediction[score_map < 0.7] = 0
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_threspred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            fakelabel=np.zeros_like(label)
            fakelabel[:,50,:]=label[:,50,:]
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
            nib.save(nib.Nifti1Image(fakelabel[:].astype(np.float32), np.eye(4)), test_save_path + id + "_fakegt.nii.gz")
    avg_dice = totaldice / len(image_list)
    avg_iou=totaliou/len(image_list)
    avg_hd=totalhd/len(image_list)
    avg_asd=totalasd/len(image_list)
    print('average metric is {},{},{},{}'.format(avg_dice,avg_iou,avg_hd,avg_asd))
    print(dicelist)
    return avg_dice,avg_iou,avg_hd,avg_asd,dicelist


def test_single_case(net1, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net1(test_patch)

                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    score_map=np.max(score_map,axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        #score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map




def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
