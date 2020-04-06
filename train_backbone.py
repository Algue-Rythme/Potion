import argparse
import collections
import csv
import os
from os import path

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm

from datamgr import SimpleDataManager, SetDataManager
from wrn import wrn28_10
from io_utils import parse_args, get_resume_file, get_assigned_file


use_gpu = torch.cuda.is_available()


def get_rotations_triplet(inputs, targets, split_ratio=None):
    batch_size = inputs.size(0)
    rotated_inputs = []
    rotated_targets = []
    angles_indexes = []
    indices = np.arange(batch_size)
    if split_ratio is not None:
        np.random.shuffle(indices)
        split_size = int(batch_size * split_ratio)
        indices = indices[:split_size]
    for j in indices:
        x90 = inputs[j].transpose(2,1).flip(1)
        x180 = x90.transpose(2,1).flip(1)
        x270 =  x180.transpose(2,1).flip(1)
        rotated_inputs += [inputs[j], x90, x180, x270]
        rotated_targets += [targets[j] for _ in range(4)]
        angles_indexes += list(range(4))
    rotated_inputs = torch.stack(rotated_inputs)
    rotated_targets = torch.stack(rotated_targets)
    angles_indexes = torch.LongTensor(angles_indexes)
    if use_gpu:
        rotated_inputs = rotated_inputs.cuda()
        rotated_targets = rotated_targets.cuda()
        angles_indexes = angles_indexes.cuda()
    return rotated_inputs, rotated_targets, angles_indexes

def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate(base_loader_test, model, rotate_classifier=None):
    model.eval()
    if rotate_classifier is not None:
        rotate_classifier.eval()
    with torch.no_grad():
        test_loss = 0
        correct, total = 0, 0
        for batch_idx, (inputs, targets) in enumerate(base_loader_test):
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            if rotate_classifier is not None:
                triplet = get_rotations_triplet(inputs, targets, split_ratio=None)
                inputs, targets, angles_indexes = triplet

            out_latent, outputs = model.forward(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if rotate_classifier is not None:
                rotate_logits = rotate_classifier(out_latent)
                rotate_predicted = torch.argmax(rotate_logits, 1)
                rotate_correct += (rotate_predicted==angles_indexes).sum().item()
        
        print("Epoch {0} : Accuracy {1}".format(epoch, (float(correct)*100)/total), end='')
        if rotate_classifier is not None:
            print('') # new line
        else:
            print(" : Rotate Accuracy {2}".format(float(rotate_correct)*100)/total)
    torch.cuda.empty_cache()  # ?


def update_batch_infos(progress, classifier_losses, rotate_losses, correct, total):
    avg_train_loss = np.mean(classifier_losses)
    avg_rot_loss = np.mean(rotate_losses)
    acc = 100.*correct/total
    desc =  'Loss: %.3f | Acc: %.3f%% | RotLoss: %.3f '%(avg_train_loss, acc, avg_rot_loss)
    progress.set_description(desc=desc)
    progress.update()


def train_s2m2(base_loader, base_loader_val, model, start_epoch, stop_epoch, params, fine_tuning):
    cross_entropy = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    # model.final_feat_dim == 640
    rotate_classifier = nn.Sequential(nn.Linear(640, 4))
    if use_gpu:
        rotate_classifier.cuda()

    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
            ])

    print("stop_epoch", start_epoch, stop_epoch)

    MixupTuple = collections.namedtuple('MixupParams', 'targets hidden input_space alpha lam')

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        rotate_classifier.train()
        classifier_losses, rotation_losses = [], []
        correct, total = 0, 0

        if use_gpu:
            torch.cuda.empty_cache()
        
        progress = tqdm.tqdm(total=len(base_loader), leave=True, desc='training')
        for batch_idx, (inputs, targets) in enumerate(base_loader):

            optimizer.zero_grad()
            
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            if fine_tuning:
                lam = np.random.beta(params.alpha, params.alpha)
                mixup_params = MixupTuple(targets=targets, hidden=True, alpha=params.alpha, lam=lam)
                _, outputs, target_a, target_b = model(inputs, mixup_params)
                loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
                classifier_losses.append(loss.item())
                loss.backward()  # Mixup loss

                _, predictions = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (lam * predictions.eq(target_a).cpu().sum().float()
                            + (1 - lam) * predictions.eq(target_b).cpu().sum().float())

            rotations_split_ratio = 0.25 if fine_tuning else 1.  # less rotations with fine tuning
            triplet = get_rotations_triplet(inputs, targets, split_ratio=rotations_split_ratio)
            inputs, targets, angles_indexes = triplet

            latent_space, outputs = model(inputs)
            rotate_outputs = rotate_classifier(latent_space)

            rotation_loss = cross_entropy(rotate_outputs, angles_indexes)
            classifier_loss = criterion(outputs, targets)  # supervised loss (softmax, cosine... )
            loss = rotation_loss + classifier_loss

            rotation_losses.append(rotation_loss.item())
            if not fine_tuning:
                _, predictions = torch.max(outputs, 1)
                correct += predictions.eq(targets).cpu().sum().float()
                total += targets.size(0)
                classifier_losses.append(classifier_loss.item())
            
            loss.backward()  # Rotation loss + Cosine Loss
            optimizer.step()

            update_batch_infos(progress, classifier_losses, rotation_losses, correct, total)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)

        if fine_tuning:
            evaluate(base_loader_val, model, rotate_classifier=None)  # do not check rotations anymore
        else:
            evaluate(base_loader_val, model, rotate_classifier=rotate_classifier)
       
    return model 


def resume_training(checkpoint_dir, model):
    resume_file = get_resume_file(params.checkpoint_dir )        
    print("resume_file", resume_file)
    tmp = torch.load(resume_file)
    start_epoch = tmp['epoch']+1
    print("restored epoch is" , tmp['epoch'])
    state = tmp['state']                 
    model.load_state_dict(state)
    return start_epoch

def enable_gpu_usage(model):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))  
    model.cuda()
    return model


if __name__ == '__main__':
    params = parse_args('train')

    image_size = 84
    # the weights are stored into ./weights folder
    save_dir = './weights'
    # the location of the json files, themselves containing the location of the images
    data_dir = {}
    data_dir['cifar']           = './filelists/cifar/' 
    data_dir['CUB']             = './filelists/CUB/' 
    data_dir['miniImagenet']    = './filelists/miniImagenet/' 

    base_file = data_dir[params.dataset] + 'base.json'
    val_file = data_dir[params.dataset] + 'val.json'
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    base_datamgr_val = SimpleDataManager(image_size, batch_size=params.test_batch_size)
    base_loader_val = base_datamgr_val.get_data_loader(val_file, aug=False)

    if params.model == 'WideResNet28_10':
        model = wrn28_10(num_classes=params.num_classes)
    else:
        raise ValueError

    if use_gpu:
        model = enable_gpu_usage(model)

    if params.resume:
        start_epoch = resume_training(params.checkpoint_dir, model)
    elif params.method =='S2M2_R':
        resume_rotate_file_dir = params.checkpoint_dir.replace("S2M2_R","rotation")
        start_epoch = resume_training(resume_rotate_file_dir, model)
    
    if params.method =='S2M2_R':
        model = train_s2m2(base_loader, base_loader_val, model, start_epoch, start_epoch+stop_epoch, params, fine_tuning=True)
    elif params.method =='rotation':
        model = train_s2m2(base_loader, base_loader_val, model, start_epoch, stop_epoch, params, fine_tuning=False)
