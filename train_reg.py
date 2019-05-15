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
from model import ModelReg as Model
# from model2 import ModelReg as Model


# hyper-parameters
init_lr = 1e-3
epoch = 128
batch_size = 16
USE_CUDA = torch.cuda.is_available()


root_dir = './data/training/pts_no_rot/'
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for i, data in enumerate(train_loader, 0):

        pts_xyz = data['pts_xyz']
        pts_label = data['pts_bbox']

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

        # Smooth L1 Loss
        loss_reg = (pts_pred - pts_label) / (torch.abs(pts_label))
        # print(loss_reg)
        target = torch.zeros(pts_pred.size())
        if USE_CUDA:
            target = target.cuda()
        loss = smooth_l1(loss_reg, target)

        # backward + optimization
        loss.backward(retain_graph=True)
        optimizer.step()

        # print loss
        running_loss = loss.item()
        print('[Epoch %4s, Step %4s] Loss: %.4f' % (e, i, running_loss))

        loss_all.append(running_loss)

    # save model
    # torch.save(model.state_dict(), 'model/model_%03d.pkl' % e)
    torch.save(model, 'model/model_%03d.pkl' % e)
    print('Finished training epoch %4s, \"model_%03d.pkl\" was saved to \"model/\"' % (e, e))
    print('\n')

loss_all = np.array(loss_all)
np.save('loss_all.npy', loss_all)


