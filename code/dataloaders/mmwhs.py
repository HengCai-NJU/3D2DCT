import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose

import pdb

class MMWHS(Dataset):
    # base_dir="../data/processed_v1_h5"
    def __init__(self, base_dir=None, split='train',  num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        if 'train' in split:
            with open('../data/' + split + '_mmwhs.txt', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + '/test_mmwhs.txt', 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f=h5py.File(self._base_dir+"/{}.h5".format(image_name),'r')
        image = h5f['image'][:]     # 64*192*192
        label = h5f['label'][:]
        image = (image - np.mean(image)) / np.std(image)

        sample = {'image': image, 'label': label.astype(np.uint8)}
        #sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample




class MMWHS_Sparse(Dataset):
    # base_dir="../data/processed_v1_h5"
    def __init__(self, base_dir=None, split='train', slice_strategy=1,num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.slice_random=np.random.randint(0,slice_strategy,16)
        print(self.slice_random)
        self.slice1=[i for i in list(range(0,96,slice_strategy))]
        self.slice2=[i for i in list(range(0,192,slice_strategy*2))]


        if 'train' in split:
            with open( '../data/'+split+'_mmwhs.txt', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + '/test_mmwhs.txt', 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f=h5py.File(self._base_dir+"/{}.h5".format(image_name),'r')
        image = h5f['image'][:]     # 64*192*192
        labeltmp = h5f['label'][:]
        label = np.zeros_like(labeltmp)
        slice1=[i+self.slice_random[idx] for i in self.slice1]
        slice2 = [i + 2*self.slice_random[idx] for i in self.slice2]
        for i in slice1:
            label[:,:,i]=labeltmp[:,:,i]
        for i in slice2:
            label[:,i,:]=labeltmp[:,i,:]
        # label[:, label.shape[1] // 2-10, :] = labeltmp[:, label.shape[1] // 2-10, :]
        # label[:, slice2, :] = labeltmp[:,slice2, :]
        # label=labeltmp
        image = (image - np.mean(image)) / np.std(image)
        sample = {'image': image, 'label': label.astype(np.uint8), 'slice1': slice1,'slice2':slice2}
        # sample = {'image': image, 'label': label, 'slice1': slice1,'idx':idx}
        if self.transform:
            sample = self.transform(sample)
        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}

class MMRandomCrop_Sparse(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label,slice1,slice2 = sample['image'], sample['label'],sample['slice1'],sample['slice2']
        #image, label, slice1 = sample['image'], sample['label'], sample['slice1']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        weight = np.zeros_like(image)
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        #weight[:]=1

        for i in slice1:
            weight[:, :, i+3] = 1
        for i in slice2:
            weight[:,i,:]=1

        weight = weight[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label, 'weight': weight}



class MMRandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        if 'weight' in sample:
            image, label,weight = sample['image'], sample['label'],sample['weight']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            weight=np.rot90(weight, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            weight=np.flip(weight, axis=axis).copy()
            return {'image': image, 'label': label,'weight':weight}
        else:
            image, label = sample['image'], sample['label']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

            return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        elif 'weight' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),'weight': torch.from_numpy(sample['weight'])}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class CenterCrop_2(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)

            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw,ph,pd=0,0,0


        (w, h, d) = label.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))
        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2] :
                x = np.pad(x, [(pw, pw), (ph, ph),(pd,pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1],d1:d1+self.output_size[2]]
            return x

        return do_transform


    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]

class ToTensor_2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1],image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]

class make_data_3d(Dataset):
    def __init__(self, imgs, plabs, masks, labs,patch_size):
        self.img = [img.squeeze() for img in imgs]
        self.plab = [np.squeeze(lab) for lab in plabs]
        self.mask = [np.squeeze(mask) for mask in masks]
        self.lab = [np.squeeze(lab) for lab in labs]
        self.num = len(self.img)
        self.tr_transform = Compose([
            CenterCrop_2(patch_size),
            # RandomNoise(),
            ToTensor_2()
        ])

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab[idx], self.mask[idx], self.lab[idx]
        # pdb.set_trace()
        samples = self.tr_transform(samples)
        imgs, plabs, masks, labs = samples
        return imgs, plabs.long(), masks.float(), labs.long()

    def __len__(self):
        return self.num
