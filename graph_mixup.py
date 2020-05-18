import collections
import os
import random

import torch
import torch.nn as nn
import tqdm

from datamgr import SetDataManager
from backbone import wrn28_10
from io_utils import parse_args, resume_training, enable_gpu_usage
from top_losses import LossesBag
from losses import get_rotations, get_bag


use_gpu = torch.cuda.is_available()

def evaluate(val_loader, model, losses_bag, params, save_latent=False):
    losses_bag.clear_epoch()
    if not params.local_batch:
        model.eval()
        losses_bag.eval()
    else:
        model.train()
        losses_bag.train()
    if save_latent:
        penultimate_dict = collections.defaultdict(list)
        features_dict = collections.defaultdict(list)
    with torch.no_grad():
        progress = tqdm.tqdm(total=len(val_loader), leave=True, ascii=True)
        for _, (inputs, targets) in enumerate(val_loader):
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs = torch.flatten(inputs, 0, 1)
            targets = torch.flatten(targets, 0, 1)

            progress_desc = []
            out_latent = model.forward(inputs)
            if params.unit_sphere:
                out_latent = normalize(out_latent)

            if params.triplet:
                _ = losses_bag['triplet'](out_latent, targets)
                progress_desc.append(losses_bag['triplet'].get_desc())

            if params.rotation:
                angles = torch.zeros([int(inputs.shape[0])], dtype=torch.int64).cuda()
                _ = losses_bag['rotation'](out_latent, angles)
                progress_desc.append(losses_bag['rotation'].get_desc())

            if params.mixup:
                _ = losses_bag['mixup'](out_latent)  # disentanglement
                progress_desc.append(losses_bag['mixup'].get_desc())

            if save_latent:
                features_latent = losses_bag.agregate_features()
                penultimate_latent = out_latent.detach().cpu().numpy()
                for penultimate, features, target in zip(penultimate_latent, features_latent, targets):
                    penultimate_dict[int(target.item())].append(penultimate)
                    features_dict[int(target.item())].append(features)

            progress_desc = ' '.join(progress_desc)
            progress.set_description(desc=progress_desc)
            progress.update()
        progress.close()
    # torch.cuda.empty_cache()  # ?
    if save_latent:
        return penultimate_dict, features_dict
    return None

def normalize(out):
    norms = torch.clamp(torch.sum(out**2, dim=1), min=1e-5)  # positive sum
    out = out / norms  # unit sphere
    return out

def train_epoch(model, losses_bag, base_loader, optimizer, params):
    model.train()
    losses_bag.train()
    losses_bag.clear_epoch()
    if use_gpu:
        torch.cuda.empty_cache()

    progress = tqdm.tqdm(total=len(base_loader), leave=True, ascii=True)
    for _, (inputs, targets) in enumerate(base_loader):
        progress_desc = []
        optimizer.zero_grad()

        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.flatten(inputs, 0, 1)
        targets = torch.flatten(targets, 0, 1)

        if params.rotation:
            inputs, angles = get_rotations(inputs)

        latent_space = model(inputs)
        if params.unit_sphere:
            latent_space = normalize(latent_space)

        if params.triplet:
            loss, _ = losses_bag['triplet'](latent_space, targets)
            loss.backward(retain_graph=True)
            progress_desc.append(losses_bag['triplet'].get_desc())

        if params.rotation:
            loss, _ = losses_bag['rotation'](latent_space, angles)
            loss.backward(retain_graph=True)
            progress_desc.append(losses_bag['rotation'].get_desc())

        if params.mixup:
            loss, _ = losses_bag['mixup'](latent_space)
            loss.backward(retain_graph=False)  # clear yo mama
            progress_desc.append(losses_bag['mixup'].get_desc())

        optimizer.step()

        progress_desc = ' '.join(progress_desc)
        progress.set_description(desc=progress_desc)
        progress.update()
    progress.close()

def full_training(base_loader, val_loader, model, start_epoch, stop_epoch, params):
    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                ] + losses_bag.optimizer_dict())

    print("stop_epoch", start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        train_epoch(model, losses_bag, base_loader, optimizer, params)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            model_dict = {'epoch':epoch, 'state':model.state_dict(), **losses_bag.states_dict()}
            torch.save(model_dict, outfile)

        evaluate(val_loader, model, losses_bag, params)

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
    base_datamgr = SetDataManager(base_file, image_size, n_way, n_shot, n_val)
    val_datamgr = SetDataManager(val_file, image_size, n_way, n_shot, n_val)
    base_loader = base_datamgr.get_data_loader(aug=params.train_aug)
    val_loader = val_datamgr.get_data_loader(aug=False)

    if params.model == 'WideResNet28_10':
        model = wrn28_10(num_classes=params.num_classes)
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError

    bag = get_bag(params)
    losses_bag = LossesBag(bag)

    if use_gpu:
        model = enable_gpu_usage(model)
        losses_bag.use_gpu()

    if params.resume:
        start_epoch = resume_training(params.checkpoint_dir, model)
        losses_bag.load_states(params.checkpoint_dir)

    model = full_training(base_loader, val_loader, model, start_epoch, start_epoch+stop_epoch, params)
