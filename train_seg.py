# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import os
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data_loader import MyDataset
# from model import ModelSeg as Model
from model2 import PointNet2SemSeg as Model


# hyper-parameters
init_lr = 1e-2
epoch = 128
batch_size = 4
USE_CUDA = torch.cuda.is_available()


root_dir = './data/pts_no_rot/'
fn_list = glob.glob(root_dir + '*.csv')
fn_list.sort()
ids = [os.path.basename(fn).split('.')[0] for fn in fn_list]

my_dataset = MyDataset(root_dir=root_dir,
                       ids=ids)

# sample = my_dataset[0]
# print(sample['pts_xyz'], sample['pts_label'], sample['pts_bbox'])

train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

# network
model = Model().train()
if USE_CUDA:
    model = model.cuda()

# criterion
smooth_l1 = nn.SmoothL1Loss()
cross_entropy = nn.CrossEntropyLoss()
nllloss = nn.NLLLoss()

# training
loss_all = []
for e in range(epoch):

    lr = init_lr / np.power(2, (e // 8))

    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for i, data in enumerate(train_loader, 0):

        pts_xyz = data['pts_xyz']
        pts_label = data['pts_label']

        pts_xyz = pts_xyz.float()
        pts_label = pts_label.float()

        # GPU
        if USE_CUDA:
            pts_xyz = pts_xyz.cuda()
            pts_label = pts_label.cuda()

        # zero grade
        optimizer.zero_grad()

        # forword
        pts_pred = model(pts_xyz)

        # Cross entropy Loss
        pts_label = pts_label.long()
        # loss = cross_entropy(pts_pred, pts_label)
        loss = nllloss(pts_pred, pts_label)
        # loss[:, 0, :] *= 1  # weighted

        # backward + optimization
        loss.backward(retain_graph=True)
        optimizer.step()

        # print loss
        running_loss = loss.item()
        print('[Epoch %4s, Step %4s] Loss: %.3f' % (e, i, running_loss))

        loss_all.append(running_loss)

    # save model
    # torch.save(model.state_dict(), 'model/model_%03d.pkl' % e)
    torch.save(model, 'model/model_%03d.pkl' % e)
    print('Finished training epoch %4s, \"model_%03d.pkl\" was saved to \"model/\"' % (e, e))
    print('\n')

loss_all = np.array(loss_all)
np.save('loss_all.npy', loss_all)


