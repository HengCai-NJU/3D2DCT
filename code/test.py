import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
parser.add_argument('--dataset',type=str,default="la",help='dataset to use')
parser.add_argument('--modeleffe',type=int,default=1,help='model to use')
parser.add_argument('--mid_iterations', type=int,  default=6000)
parser.add_argument('--max_iteration', type=int,  default=2000)
parser.add_argument('--iteration_step', type=int,  default=100)
parser.add_argument('--split', type=str,  default='test', help='testlist to use')
parser.add_argument('--min_iteration', type=int,  default=100)
FLAGS = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
import torch
from networks.vnet import VNet

from test_util import test_all_case



snapshot_path = "../../../../data/xx/3d2d/model/"+FLAGS.model+"/"
test_save_path = "../../../../data/xx/3d2d/model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)


if FLAGS.dataset=='mmwhs':
    num_classes=8
    with open('../data/'+FLAGS.split+'_mmwhs.txt', 'r') as f:
       image_list = f.readlines()
    image_list = ['../../../../data/caiheng/sas/data/MMWHS/'+item.replace('\n', '').split(",")[0]+'.h5' for item in image_list]



def test_calculate_metric(epoch_num):
    net1 = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path1 = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    print(save_mode_path1)

    #for param in net2.block_one.parameters():
    #    print(param)
    print("init weight from {}".format(save_mode_path1))
    net1.eval()

    if FLAGS.dataset=='mmwhs':
        net1.load_state_dict(torch.load(save_mode_path1), strict=True)
        avg_dice,avg_iou,avg_hd,avg_asd,dice_list = test_all_case(net1, image_list, num_classes=num_classes,
                                   patch_size=(192, 192, 96), stride_xy=16, stride_z=4,
                                   save_result=True, test_save_path=test_save_path, dataset=FLAGS.dataset)


    return avg_dice,avg_iou,avg_hd,avg_asd,dice_list


if __name__ == '__main__':
    maxmetric=0
    maxi=-1
    path=os.path.join(snapshot_path, 'test.txt')
    for i in range(FLAGS.min_iteration,FLAGS.max_iteration+1,FLAGS.iteration_step):
        avg_dice,avg_iou,avg_hd,avg_asd,dice_list=test_calculate_metric(i)
        strmetric = 'net'+str(FLAGS.modeleffe)+": iter"+str(i)+":\n"+str(avg_dice)+'\n'+str(avg_iou)+'\n'+str(avg_hd)+'\n'+str(avg_asd)+'\n'
        with open (path,"a") as f:
            f.writelines(strmetric)
        if avg_dice[-1]>maxmetric:
            maxi=i
            maxmetric=avg_dice[-1]
    print(maxmetric,"||",maxi)
    with open(path, "a") as f:
        f.writelines(str(maxi)+'\n')
    with open('../data/'+(FLAGS.split).replace('valid','test')+'_mmwhs.txt', 'r') as f:
       image_list = f.readlines()
    image_list = ['../../../../data/xx/3d2d/data/MMWHS/'+item.replace('\n', '').split(",")[0]+'.h5' for item in image_list]
    avg_dice,avg_iou,avg_hd,avg_asd,dice_list=test_calculate_metric(maxi)
    strmetric = 'net' + str(FLAGS.modeleffe) + ": iter" + str(maxi) + ":\n" + str(avg_dice) + '\n' + str(
        avg_iou) + '\n' + str(avg_hd) + '\n' + str(avg_asd) + '\n'
    with open(path, "a") as f:
        f.writelines(strmetric)