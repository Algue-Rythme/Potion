import os
import random

import torch
import torch.nn as nn
import tqdm

from datamgr import SetDataManager
from backbone import wrn28_10
from io_utils import parse_args, resume_training, enable_gpu_usage
from top_losses import LossesBag
from losses import RotationLoss, TripletLoss, get_rotations, MixupLoss


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

            _ = losses_bag['triplet'](out_latent, targets)
            progress_desc = [losses_bag['triplet'].get_desc()]

            angles = torch.zeros([int(inputs.shape[0])], dtype=torch.int64).cuda()
            _ = losses_bag['rotations'](out_latent, angles)
            progress_desc.append(losses_bag['rotations'].get_desc())

            _ = losses_bag['mixup'](out_latent)
            progress_desc.append(losses_bag['mixup'].get_desc())

            progress_desc = ' '.join(progress_desc)
            progress.set_description(desc=progress_desc)
            progress.update()
        progress.close()
    torch.cuda.empty_cache()  # ?

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
        inputs, angles = get_rotations(inputs)

        latent_space = model(inputs)
        if params.unit_sphere:
            latent_space = normalize(latent_space)

        loss, _ = losses_bag['triplet'](latent_space, targets)
        loss.backward(retain_graph=True)
        progress_desc.append(losses_bag['triplet'].get_desc())

        loss, _ = losses_bag['rotation'](latent_space, angles)
        loss.backward(retain_graph=True)
        progress_desc.append(losses_bag['rotation'].get_desc())

        loss, _ = losses_bag['mixup'](latent_space)
        loss.backward(retain_graph=False)  # clear yo mama
        progress_desc.append(losses_bag['mixup'].get_desc())

        optimizer.step()

        progress_desc = ' '.join(progress_desc)
        progress.set_description(desc=progress_desc)
        progress.update()
    progress.close()

def full_training(base_loader, base_loader_val, model, start_epoch, stop_epoch, params):
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

        evaluate(base_loader_val, model, losses_bag)

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
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError

    losses_bag = LossesBag([
        TripletLoss(640, 128, 64, params.n_way),
        RotationLoss(640, 128),
        MixupLoss(640, 128, beta_param=0.4)
        ])

    if use_gpu:
        model = enable_gpu_usage(model)
        losses_bag.use_gpu()

    if params.resume:
        start_epoch = resume_training(params.checkpoint_dir, model)
        losses_bag.load_states(params.checkpoint_dir)

    model = full_training(base_loader, base_loader_val, model, start_epoch, start_epoch+stop_epoch, params)
