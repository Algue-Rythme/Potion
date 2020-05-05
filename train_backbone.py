import collections
import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm

from datamgr import SimpleDataManager
from wrn import wrn28_10
from io_utils import parse_args, get_resume_file


use_gpu = torch.cuda.is_available()


def get_rotations_triplet(inputs, targets, rotation_type=None):
    batch_size = inputs.size(0)
    if rotation_type is None:
        return inputs, targets, torch.LongTensor([0]*batch_size)
    rotated_inputs = []
    rotated_targets = []
    angles_indexes = []
    if rotation_type == 'some':
        indices = [[random.randrange(4)] for _ in range(batch_size)]
    elif rotation_type == 'all':
        indices = [list(range(4)) for _ in range(batch_size)]
    for j in range(batch_size):
        x90 = inputs[j].transpose(2,1).flip(1)
        x180 = x90.transpose(2,1).flip(1)
        x270 =  x180.transpose(2,1).flip(1)
        rotations = [inputs[j], x90, x180, x270]
        rotations = [rotations[i] for i in indices[j]]
        rotated_inputs += rotations
        rotated_targets += [targets[j] for _ in range(len(rotations))]
        angles_indexes += indices[j]
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

def evaluate(base_loader_val, model, rotate_classifier=None):
    cross_entropy = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    model.eval()
    classifier_losses = []
    classifier_acc = 0, 0
    if rotate_classifier is not None:
        rotate_classifier.eval()
        rotation_losses = []
        rotations_acc = 0, 0

    with torch.no_grad():
        progress = tqdm.tqdm(total=len(base_loader_val), leave=True, ascii=True)
        for _, (inputs, targets) in enumerate(base_loader_val):
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            if rotate_classifier is not None:
                inputs, targets, angles_indexes = get_rotations_triplet(inputs, targets, rotation_type='some')

            out_latent, outputs = model.forward(inputs)

            classifier_loss = criterion(outputs, targets)  # supervised loss (softmax, cosine... )
            classifier_losses.append(classifier_loss.item())
            classifier_acc = update_acc(classifier_acc, outputs, targets)
            desc = metric_desc('ce', classifier_losses, *classifier_acc)

            if rotate_classifier is not None:
                rotate_logits = rotate_classifier(out_latent)
                rotation_loss = cross_entropy(rotate_logits, angles_indexes)
                rotation_losses.append(rotation_loss.item())
                rotations_acc = update_acc(rotations_acc, rotate_logits, angles_indexes)
                desc += metric_desc('rot', rotation_losses, *rotations_acc)

            progress.set_description(desc=desc)
            progress.update()
        progress.close()
    torch.cuda.empty_cache()  # ?

def metric_desc(name, losses, correct_total=None, total=None):
    avg_loss = np.mean(losses)
    desc = ' '+name+('_loss=%.3f'%avg_loss)
    if correct_total is not None:
        correct, total = correct_total
        acc = 100.*correct/total
        desc += ' '+name+('_acc=%.2f%%'%acc)
    return desc

def update_acc(correct_total, outputs, targets):
    _, predictions = torch.max(outputs, 1)
    correct, total = correct_total
    correct += predictions.eq(targets).cpu().sum().item()
    total += targets.size(0)
    return correct, total

def train_epoch(model, rotate_classifier, base_loader, optimizer, fine_tuning):
    cross_entropy = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    MixupTuple = collections.namedtuple('MixupParams', 'targets hidden input_space alpha lam')

    model.train()
    rotate_classifier.train()
    classifier_losses, rotation_losses = [], []
    classifier_acc = 0, 0

    if use_gpu:
        torch.cuda.empty_cache()

    progress = tqdm.tqdm(total=len(base_loader), leave=True, ascii=True)
    for _, (inputs, targets) in enumerate(base_loader):
        desc = ''

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

        inputs, targets, angles_indexes = get_rotations_triplet(inputs, targets, rotation_type='some')
        latent_space, outputs = model(inputs)

        classifier_loss = criterion(outputs, targets)  # supervised loss (softmax, cosine... )
        classifier_losses.append(classifier_loss.item())
        classifier_acc = update_acc(classifier_acc, outputs, targets)
        desc += metric_desc('ce', classifier_losses, classifier_acc)

        rotate_outputs = rotate_classifier(latent_space)
        rotation_loss = cross_entropy(rotate_outputs, angles_indexes)
        rotation_losses.append(rotation_loss.item())
        desc += metric_desc('rot', rotation_losses)

        loss = rotation_loss + classifier_loss
        loss.backward()
        optimizer.step()

        progress.set_description(desc=desc)
        progress.update()

    progress.close()


def train_s2m2(base_loader, base_loader_val, model, start_epoch, stop_epoch, params, fine_tuning):
    final_feat_dim = 640  # model.final_feat_dim
    rotate_classifier = nn.Sequential(nn.Linear(final_feat_dim, 4))
    if use_gpu:
        rotate_classifier.cuda()

    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
                ])

    print("stop_epoch", start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        train_epoch(model, rotate_classifier, base_loader, optimizer, fine_tuning)

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
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s/%s' %(save_dir, params.dataset, params.model, params.method, params.run_name)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug, num_workers=16, lazy_load=params.lazy_load)
    base_datamgr_val = SimpleDataManager(image_size, batch_size=params.test_batch_size)
    base_loader_val = base_datamgr_val.get_data_loader(val_file, aug=False, num_workers=16, lazy_load=params.lazy_load)

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
