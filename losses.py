import random
import torch
import torch.nn as nn
from top_losses import LossEngine


use_gpu = torch.cuda.is_available()


def get_pos_neg(x_latent, n_way):
    batch_size = x_latent.shape[0]
    per_class = int(batch_size / n_way)
    indexes = list(range(batch_size))
    positives = [random.sample(indexes[per_class*i:per_class*(i+1)], per_class) for i in range(n_way)]
    positives = [p for pos in positives for p in pos]
    negatives = [random.sample(indexes[:per_class*i]+indexes[per_class*(i+1):], per_class) for i in range(n_way)]
    negatives = [n for neg in negatives for n in neg]
    x_pos = x_latent[positives,:]
    x_neg = x_latent[negatives,:]
    return x_pos, x_neg

class DoubletLoss(LossEngine):
    def __init__(self, input_dim, intermediate_dim, n_way):
        super(DoubletLoss, self).__init__(name='doublet', accuracy=True)
        self.n_way = n_way
        self.lin1 = nn.Linear(input_dim, intermediate_dim)
        self.act1 = nn.SELU()
        self.lin2 = nn.Linear(intermediate_dim, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def get_features(self, x_latent):
        x_latent = self.lin1(x_latent)
        x_latent = self.act1(x_latent)
        return x_latent

    def forward(self, x_latent, _):
        x_latent = self.get_features(x_latent)
        batch_size = x_latent.shape[0]
        x_pos, x_neg = get_pos_neg(x_latent, self.n_way)
        pos_scores = torch.squeeze(self.lin2(x_latent * x_pos))
        neg_scores = torch.squeeze(self.lin2(x_latent * x_neg))
        ones = torch.ones([batch_size], dtype=torch.int64)
        zeros = torch.zeros([batch_size], dtype=torch.int64)
        if use_gpu:
            ones, zeros = ones.cuda(), zeros.cuda()
        pos_loss = self.bce_loss(pos_scores, ones)
        neg_loss = self.bce_loss(neg_scores, zeros)
        loss = pos_loss + neg_loss
        self.losses_items.append(float(loss.item()))
        self.update_acc(pos_scores, ones)
        self.update_acc(neg_scores, zeros)
        return loss, x_latent


class TripletLoss(LossEngine):
    def __init__(self, input_dim, intermediate_dim, final_dim, n_way):
        super(TripletLoss, self).__init__(name='triplet', accuracy=True)
        self.n_way = n_way
        self.lin1 = nn.Linear(input_dim, intermediate_dim)
        self.act1 = nn.SELU()
        self.lin2 = nn.Linear(intermediate_dim, final_dim)
        self.act2 = nn.SELU()
        # self.lin3 = nn.Linear(intermediate_dim, final_dim)
        # self.act3 = nn.SELU()
        self.bilinear = nn.Parameter(data=torch.zeros([final_dim, intermediate_dim]),
                                     requires_grad=True)
        self.ce_loss = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.bilinear)

    def score_against(self, a_b, c):
        a, b = a_b
        ab = self.act2(self.lin2(a * b))
        # c = self.act3(self.lin3(c))
        abc = torch.einsum('bi,ij,bj->b', ab, self.bilinear, c)
        return abc  # shape B

    def get_features(self, x_latent):
        x_latent = self.lin1(x_latent)
        x_latent = self.act1(x_latent)
        return x_latent

    def forward(self, x_latent, _):
        x_latent = self.get_features(x_latent)
        x_pos, x_neg = get_pos_neg(x_latent, self.n_way)
        abc = self.score_against([x_latent, x_pos], x_neg)
        acb = self.score_against([x_latent, x_neg], x_pos)
        bca = self.score_against([x_pos, x_neg], x_latent)
        logits = torch.stack([abc, acb, bca], dim=1)
        batch_size = int(x_latent.shape[0])
        fake_target = torch.zeros([batch_size], dtype=torch.int64)
        if use_gpu:
            fake_target = fake_target.cuda()
        loss = self.ce_loss(logits, fake_target)
        self.losses_items.append(float(loss.item()))
        self.update_acc(logits, fake_target)
        return loss, x_latent

def get_rotations(inputs):
    batch_size = inputs.size(0)
    rotated_inputs = []
    angles_indexes = []
    indices = [[random.randrange(4)] for _ in range(batch_size)]
    for j in range(batch_size):
        x90 = inputs[j].transpose(2,1).flip(1)
        x180 = x90.transpose(2,1).flip(1)
        x270 =  x180.transpose(2,1).flip(1)
        rotations = [inputs[j], x90, x180, x270]
        rotations = [rotations[i] for i in indices[j]]
        rotated_inputs += rotations
        angles_indexes += indices[j]
    rotated_inputs = torch.stack(rotated_inputs)
    angles_indexes = torch.LongTensor(angles_indexes)
    if use_gpu:
        angles_indexes = angles_indexes.cuda()
    return rotated_inputs, angles_indexes

class RotationLoss(LossEngine):
    def __init__(self, input_dim, intermediate_dim):
        super(RotationLoss, self).__init__('rotation', accuracy=True)
        self.lin1 = nn.Linear(input_dim, intermediate_dim)
        self.act1 = nn.SELU()
        self.lin2 = nn.Linear(intermediate_dim, 4)
        self.ce_loss = nn.CrossEntropyLoss()

    def get_features(self, x_latent):
        x_latent = self.lin1(x_latent)
        x_latent = self.act1(x_latent)
        return x_latent

    def forward(self, x_latent, angles):
        x_latent = self.get_features(x_latent)
        logits = self.lin2(x_latent)
        loss = self.ce_loss(logits, angles)
        self.losses_items.append(float(loss.item()))
        self.update_acc(logits, angles)
        return loss, x_latent

class MixupLoss(LossEngine):
    def __init__(self, input_dim, intermediate_dim, beta_param):
        super(MixupLoss, self).__init__('mixup', accuracy=False)
        self.lin1 = nn.Linear(input_dim, intermediate_dim)
        self.act1 = nn.SELU()
        alpha, beta = beta_param, beta_param
        self.beta_distribution = torch.distributions.beta.Beta(alpha, beta)
        self.bilinear = nn.Parameter(data=torch.zeros([intermediate_dim, intermediate_dim]),
                                     requires_grad=True)
        self.ce_loss = nn.BCEWithLogitsLoss()
        nn.init.xavier_normal_(self.bilinear)

    def get_features(self, x_latent):
        x_latent = self.lin1(x_latent)
        x_latent = self.act1(x_latent)
        return x_latent

    def forward(self, x_latent):
        batch_size, latent_size = int(x_latent.shape[0]), int(x_latent.shape[1])
        bernouilli_p = self.beta_distribution.sample([batch_size])
        target = (bernouilli_p > 0.5).float()
        bernouilli_d = torch.distributions.Bernoulli(probs=bernouilli_p)
        bernouilli = torch.transpose(bernouilli_d.sample([latent_size]), 0, 1)
        if use_gpu:
            target = target.cuda()
            bernouilli = bernouilli.cuda()
        x_permuted = x_latent[torch.randperm(batch_size)]
        mixed = bernouilli * x_latent + (1. - bernouilli) * x_permuted  # hypercube interpolation
        mixed = self.get_features(mixed)
        x_latent = self.get_features(x_latent)
        x_permuted = self.get_features(x_permuted)
        against_latent = torch.einsum('bi,ij,bj->b', mixed, self.bilinear, x_latent)
        against_permuted = torch.einsum('bi,ij,bj->b', mixed, self.bilinear, x_permuted)
        logits = against_latent + against_permuted
        loss = self.ce_loss(logits, target)
        self.losses_items.append(float(loss.item()))
        self.update_acc(logits, target)
        return loss, x_latent
