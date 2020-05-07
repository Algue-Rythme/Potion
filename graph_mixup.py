import collections
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm

from datamgr import SetDataManager
from backbone import wrn28_10
from io_utils import parse_args, get_resume_file
from top_losses import LossEngine, LossesBag

use_gpu = torch.cuda.is_available()

def evaluate(base_loader_val, model, losses_bag):
    model.eval()
    losses_bag.eval()
    losses_bag.clear_epoch()
    with torch.no_grad():
        progress = tqdm.tqdm(total=len(base_loader_val), leave=True, ascii=True)
        for _, (inputs, targets) in enumerate(base_loader_val):
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs = torch.flatten(inputs, 0, 1)
            out_latent = model.forward(inputs)
            progress_desc = ' '.join([desc for _, _, desc in losses_bag.get_losses(out_latent, targets)])
            progress.set_description(desc=progress_desc)
            progress.update()
        progress.close()
    torch.cuda.empty_cache()  # ?

def train_epoch(model, losses_bag, base_loader, optimizer):
    model.train()
    losses_bag.train()
    losses_bag.clear_epoch()
    if use_gpu:
        torch.cuda.empty_cache()

    progress = tqdm.tqdm(total=len(base_loader), leave=True, ascii=True)
    for _, (inputs, targets) in enumerate(base_loader):
        progress_desc = ''
        optimizer.zero_grad()
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.flatten(inputs, 0, 1)
        latent_space = model(inputs)
        for _, loss, desc in losses_bag.get_losses(latent_space, targets):
            progress_desc += desc
            loss.backward()
        optimizer.step()
        progress.set_description(desc=progress_desc)
        progress.update()
    progress.close()

class TargetLoss(LossEngine):
    def __init__(self, input_dim, intermediate_dim, n_way):
        super(TargetLoss, self).__init__(name='Target', accuracy=True)
        self.n_way = n_way
        self.lin1 = nn.Linear(input_dim, intermediate_dim)
        self.act1 = nn.SELU()
        self.lin2 = nn.Linear(intermediate_dim, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x_latent, _):
        x_latent = self.lin1(x_latent)
        x_latent = self.act1(x_latent)
        batch_size = x_latent.shape[0]
        per_class = int(batch_size / n_way)
        indexes = list(range(batch_size))
        positives = [random.sample(indexes[per_class*i:per_class*(i+1)], per_class) for i in range(n_way)]
        positives = [p for pos in positives for p in pos]
        negatives = [random.sample(indexes[:per_class*i]+indexes[per_class*(i+1):], per_class) for i in range(n_way)]
        negatives = [n for neg in negatives for n in neg]
        x_pos = x_latent[positives,:]
        x_neg = x_latent[negatives,:]
        pos_scores = torch.squeeze(self.lin2(x_latent * x_pos))
        neg_scores = torch.squeeze(self.lin2(x_latent * x_neg))
        ones, zeros = torch.ones([batch_size]), torch.zeros([batch_size])
        if use_gpu:
            ones, zeros = ones.cuda(), zeros.cuda()
        pos_loss = self.bce_loss(pos_scores, ones)
        neg_loss = self.bce_loss(neg_scores, zeros)
        loss = pos_loss + neg_loss
        self.losses_items.append(float(loss.item()))
        self.update_acc(pos_scores, ones)
        self.update_acc(neg_scores, zeros)
        return loss


def full_training(base_loader, base_loader_val, model, start_epoch, stop_epoch, params):

    losses_bag = LossesBag([TargetLoss(640, 320, params.n_way)])
    if use_gpu:
        losses_bag.use_gpu()

    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                ] + losses_bag.optimizer_dict())

    print("stop_epoch", start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        train_epoch(model, losses_bag, base_loader, optimizer)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)

        evaluate(base_loader_val, model, losses_bag)

    return model

def resume_training(checkpoint_dir, model):
    resume_file = get_resume_file(checkpoint_dir)
    print("resume_file", resume_file)
    tmp = torch.load(resume_file)
    start_epoch = tmp['epoch']+1
    print("restored epoch is" , tmp['epoch'])
    state = tmp['state']
    model.load_state_dict(state)
    return start_epoch

def enable_gpu_usage(model):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    return model

if __name__ == '__main__':
    params = parse_args('graph')
    random.seed(914637)

    image_size = 84  # small
    # the weights are stored into ./weights folder
    save_dir = './weights'
    # the location of the json files, themselves containing the location of the images
    data_dir = {}
    data_dir['cifar']           = './filelists/cifar/'
    data_dir['CUB']             = './filelists/CUB/'
    data_dir['miniImagenet']    = './filelists/miniImagenet/'

    base_file = data_dir[params.dataset] + 'base.json'
    val_file = data_dir[params.dataset] + 'val.json'
    params.checkpoint_dir = '%s/checkpoints/%s/%s/%s' %(save_dir, params.dataset, params.model, params.run_name)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    n_way, n_shot, n_val = params.n_way, params.n_shot, params.n_val
    n_episode = int(600 / (n_way*(n_shot + n_val)))
    base_datamgr = SetDataManager(base_file, image_size, n_way, n_shot, n_val, n_episode * 64)
    val_datamgr = SetDataManager(base_file, image_size, n_way, n_shot, n_val, n_episode * 16)
    base_loader = base_datamgr.get_data_loader(aug=params.train_aug, num_workers=12)
    base_loader_val = val_datamgr.get_data_loader(aug=False, num_workers=12)

    if params.model == 'WideResNet28_10':
        model = wrn28_10(num_classes=params.num_classes)
    else:
        raise ValueError

    if use_gpu:
        model = enable_gpu_usage(model)

    if params.resume:
        start_epoch = resume_training(params.checkpoint_dir, model)

    model = full_training(base_loader, base_loader_val, model, start_epoch, start_epoch+stop_epoch, params)
