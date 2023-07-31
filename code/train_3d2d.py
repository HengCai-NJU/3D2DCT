import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='MT31', help='model_name')
parser.add_argument('--dataset', type=str, default='effe', help='dataset to use')
parser.add_argument('--label_num', type=int, default=5, help='number of labeled data')

parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--sliceseed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--split', type=str,  default='train', help='datalist to use')
parser.add_argument('--num', type=int,  default=16, help='number of labeled volume')
parser.add_argument('--quality_bar', type=float,  default=0.98, help='quality bar')
parser.add_argument('--ht', type=float,  default=0.9, help='hard threshold')
parser.add_argument('--st', type=float,  default=0.7, help='soft threshold')

### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--slice_strategy', type=int, default=0, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
args = parser.parse_args()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil

import logging
import time
import random
import numpy as np
import pdb

import torch
import torch.optim as optim
from torchvision import transforms
from torch.nn import DataParallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from networks.unet import UNet
gpus=[0,1,2]


from dataloaders import utils
from utils import ramps, losses

from dataloaders.mmwhs import MMWHS_Sparse, MMRandomCrop_Sparse,MMRandomRotFlip,ToTensor, TwoStreamBatchSampler





train_data_path = args.root_path
snapshot_path = "../../../../data/xx/3d2dct/model/" + args.exp + "/"
if args.dataset=='mmwhs':
    train_data_path = "../../../../data/caiheng/sas/data/MMWHS"


batch_size = 1
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.sliceseed)
    if args.dataset == 'mmwhs':
        num_classes = 8
        patch_size = (176, 176, 96)
        db_train1 = MMWHS_Sparse(base_dir=train_data_path,
                                    split=args.split,
                                    num=args.num,
                                    slice_strategy=args.slice_strategy,
                                    transform=transforms.Compose([
                                        MMRandomCrop_Sparse(patch_size),
                                        #MMRandomRotFlip(),
                                        ToTensor(),
                                    ]))

    np.random.seed(args.seed)




    def create_3dmodel(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        return model

    def create_2dmodel(ema=False):
        # Network definition
        net = UNet(n_channels=1, n_classes=num_classes)
        model = net.cuda()
        return model




    model1 = create_3dmodel()
    model1=DataParallel(model1,device_ids=gpus,output_device=gpus[0])
    model2 = create_2dmodel()
    model2=DataParallel(model2,device_ids=gpus,output_device=gpus[0])
    model3=create_2dmodel()
    model3=DataParallel(model3,device_ids=gpus,output_device=gpus[0])



    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader1 = DataLoader(db_train1, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)


    model1.train()
    model2.train()
    model3.train()
    #optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer1 = optim.Adam(model1.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    optimizer2 = optim.Adam(model2.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    optimizer3 = optim.Adam(model3.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader1)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader1) + 1
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch1 in enumerate(trainloader1):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))

            volume_batch1, label_batch1, maskz1 = sampled_batch1['image'], sampled_batch1['label'], sampled_batch1['weight']

            maskz1 = maskz1.cuda()

            maskzz1 = torch.unsqueeze(maskz1, 1).cuda()
            maskzz1 = maskzz1.repeat(1, 1, 1, 1, 1).cuda()
            volume_batch1, label_batch1 = volume_batch1.cuda(), label_batch1.cuda()
            outputs1 = model1(volume_batch1)
            outputs_soft1 = F.softmax(outputs1, dim=1)

            volume_batch2 = volume_batch1[0].transpose(0, 3).squeeze().unsqueeze(1) #[1,1,192,192,96]->[96,1,192,192]
            #print(torch.sum(outputs_soft1[:,1]>0.5))
            outputs2 = model2(volume_batch2)  # [96,8,192,192]
            outputs_soft2 = F.softmax(outputs2, dim=1)
            outputs2 = outputs2.unsqueeze(4).transpose(0, 4) #[1,8,192,192,96]
            outputs_soft2 = outputs_soft2.unsqueeze(4).transpose(0, 4)

            volume_batch3=volume_batch1[0].transpose(0,2).squeeze().unsqueeze(1)#[192,1,192,96]
            outputs3=model3(volume_batch3) #[192,8,192,96]
            outputs_soft3=F.softmax(outputs3,dim=1)
            outputs3=outputs3.unsqueeze(3).transpose(0,3)
            outputs_soft3=outputs_soft3.unsqueeze(3).transpose(0,3)

            twodthreshold=0.0
            twod1=torch.argmax(outputs_soft2.detach(), dim=1, keepdim=False)
            confidence2d1,_=torch.max(outputs_soft2.detach(), dim=1, keepdim=False)
            twod2=torch.argmax(outputs_soft3.detach(),dim=1,keepdim=False)
            confidence2d2,_=torch.max(outputs_soft3.detach(), dim=1, keepdim=False)
            threed = torch.argmax(outputs_soft1.detach(), dim=1, keepdim=False)
            confidence3d,_ = torch.max(outputs_soft1.detach(), dim=1, keepdim=False)
            threedcorrection=(twod1!=threed)*(confidence3d>confidence2d1)*(confidence3d>confidence2d2)
            threedcorrection=~threedcorrection
            twodmask=(twod1==twod2)*threedcorrection*(confidence2d1>twodthreshold)*(confidence2d2>twodthreshold)

            hardthreedthreshold = args.ht
            softthreedthreshold = args.st

            threedmask = confidence3d > hardthreedthreshold
            twodcorrection1 = (confidence2d1 > confidence3d) * (twod1 != threed)
            twodcorrection1 = ~twodcorrection1
            threedmask1 = threedmask
            twodcorrection2 = (confidence2d2 > confidence3d) * (twod2 != threed)
            twodcorrection2 = ~twodcorrection2
            threedmask2 = threedmask
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # 3d quality verification and good sample selection
            print(threed.shape)
            print(label_batch1.shape)
            print(np.count_nonzero(threed[maskz1==1]==label_batch1[maskz1==1]),np.count_nonzero(maskz1==1))
            quality=np.count_nonzero(threed[maskz1==1]==label_batch1[maskz1==1])/np.count_nonzero(maskz1==1)
            if quality>args.quality_bar:
                threedmask1=confidence3d > softthreedthreshold
                threedmask2=confidence3d > softthreedthreshold

            ## calculate the loss
            twod1[maskz1==1]=label_batch1[maskz1==1]
            twodmask=consistency_weight*twodmask
            twodmask[maskz1==1]=1
            loss_seg1 = losses.wce(outputs1, twod1, twodmask, batch_size, patch_size[0], patch_size[1],
                                      patch_size[2])
            loss_seg_dice1=losses.multi_dice_loss_weight(outputs_soft1,twod1,twodmask,classnum=7)



            supervised_loss1 = 0.5 * (loss_seg1 + loss_seg_dice1)


            threed[maskz1 == 1] = label_batch1[maskz1 == 1]
            threedmask1 = consistency_weight * threedmask1
            threedmask1[maskz1 == 1] = 1

            loss_seg2 = losses.wce(outputs2, threed, threedmask1, batch_size, patch_size[0], patch_size[1],
                                   patch_size[2])
            loss_seg_dice2= losses.multi_dice_loss_weight(outputs_soft2, threed, threedmask1, classnum=7)



            supervised_loss2 = 0.5 * (loss_seg2 + loss_seg_dice2)

            threedmask2 = consistency_weight * threedmask2
            threedmask2[maskz1 == 1] = 1
            loss_seg3 = losses.wce(outputs3, threed, threedmask2, batch_size, patch_size[0], patch_size[1],
                                   patch_size[2])

            loss_seg_dice3= losses.multi_dice_loss_weight(outputs_soft3, threed, threedmask2, classnum=7)



            supervised_loss3 = 0.5 * (loss_seg3 + loss_seg_dice3)



            # total loss
            loss = supervised_loss1+supervised_loss2+supervised_loss3

            #loss=loss_seg_dice1+loss_seg_dice2
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()



            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            logging.info('iteration %d : 3d loss : %f 2d loss : %f, %f, mask num %d %d %d, quality %f ' %(iter_num,supervised_loss1.item(),supervised_loss2.item(),supervised_loss3.item(),torch.count_nonzero(twodmask).item(),torch.count_nonzero(threedmask1).item(),torch.count_nonzero(threedmask2).item(),quality))


            if iter_num % 100 == 0 and iter_num<=6000:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model1.module.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model1.module.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
