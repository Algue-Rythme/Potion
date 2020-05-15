import collections
import os
import functools
import json
import random
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def memoize(func):
    cache = func.cache = {}
    @functools.wraps(func)
    def memoized_func(*l_args):
        key = tuple(l_args)
        if key not in cache:
            cache[key] = func(*l_args)
        return cache[key]
    return memoized_func

identity = lambda x:x

@memoize
def lazy_read_image(image_path):
    return Image.open(image_path).convert('RGB')

def read_image(image_path):
    return Image.open(image_path).convert('RGB')

class SimpleDataset:
    def __init__(self, data_file, transform, indexes=None, target_transform=identity, lazy_load=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.image_reader = lazy_read_image if lazy_load else read_image
        self.indexes = list(range(len(self.meta['image_names']))) if indexes is None else indexes

    def __getitem__(self, i):
        index = self.indexes[i]
        image_path = os.path.join(self.meta['image_names'][index])
        img = self.image_reader(image_path)
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][index])
        return img, target

    def __len__(self):
        return len(self.indexes)


class SetDataset:
    def __init__(self, data_file, batch_size, transform, balanced=True, lazy_load=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.batch_size = batch_size
        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.sub_meta = collections.defaultdict(list)

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        if balanced:
            for sub in self.sub_meta:
                cropped = len(self.sub_meta[sub]) - (len(self.sub_meta[sub]) % batch_size)
                self.sub_meta[sub] = self.sub_meta[sub][:cropped]
        else:
            for sub in self.sub_meta:
                assert len(self.sub_meta[sub]) % batch_size == 0

        self.sub_dataloader = []
        self.iterators = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0, #use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform, lazy_load=lazy_load)
            sub_dataloader = torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)
            self.sub_dataloader.append(sub_dataloader)
            self.iterators.append(iter(sub_dataloader))

    def get_classes_length(self):
        return len(self.sub_dataloader[0])  # assume same size for each class

    def __getitem__(self,i):
        try:
            return next(self.iterators[i])
        except StopIteration:
            self.iterators[i] = iter(self.sub_dataloader[i])
            return next(self.iterators[i])

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, lazy_load=False):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.image_reader = lazy_read_image if lazy_load else read_image

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join(self.sub_meta[i])
        img = self.image_reader(image_path)
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler:
    def __init__(self, n_classes, n_way, n_subbatch):
        if n_classes % n_way:
            print('WARNING n_classes (%d) %% n_way (%d) != 0 (=%d)'%(n_classes, n_way, n_classes % n_way))
            n_classes -= (n_classes%n_way)
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_subbatch = n_subbatch
        self.total_batchs = self.n_classes * self.n_subbatch

    def __len__(self):
        return self.total_batchs // self.n_way

    def __iter__(self):
        still_here = set(range(self.n_classes))
        for _ in range(len(self)):
            indexes = random.sample(still_here, self.n_way)
            still_here = still_here - set(indexes)
            if not still_here:
                still_here = set(range(self.n_classes))  # reset classes
            yield indexes
