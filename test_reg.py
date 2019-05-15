# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import os
import glob
import numpy as np

from pyquaternion import Quaternion
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data_loader import MyDataset


# hyper-parameters
batch_size = 1
USE_CUDA = torch.cuda.is_available()


root_dir = './data/test/'
fn_list = glob.glob(root_dir + '*.csv')
fn_list.sort()
ids = [os.path.basename(fn).split('.')[0] for fn in fn_list]

my_dataset = MyDataset(root_dir=root_dir,
                       ids=ids)

# sample = my_dataset[0]
# print(sample['pts_xyz'], sample['pts_label'], sample['pts_bbox'])

# test loader
test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=False)

# load trained model
model = torch.load('model/model_127.pkl', map_location='cpu').eval()

if USE_CUDA:
    model = model.cuda()

for i, data in enumerate(test_loader, 0):

    if i in range(5):

        pts_xyz = data['pts_xyz']
        pts_label = data['pts_bbox']

        pts_xyz = pts_xyz.float()
        pts_label = pts_label.float()

        if USE_CUDA:
            pts_xyz = pts_xyz.cuda()
            pts_label = pts_label.cuda()

        pts_pred = model(pts_xyz)

        pts_pred = np.array(pts_pred.detach().squeeze())
        pts_label = np.array(pts_label.detach().squeeze())

        print(pts_pred)
        print(pts_label)
        print((pts_pred - pts_label) / np.abs(pts_label))
        print('\n')

        # find points in bbox
        pts_all = data['pts_all']
        pts_all = np.array(pts_all.detach().squeeze())
        # # un-rotate
        # pts_all[:, 0:3] = np.dot(pts_all[:, 0:3], Quaternion(pts_pred[6:10]).rotation_matrix)
        # filter
        scale_factor = 1.2  # scale factor for bbox
        range1 = pts_pred[0:3] - scale_factor * pts_pred[3:6] / 2
        range2 = pts_pred[0:3] + scale_factor * pts_pred[3:6] / 2
        idx = (pts_all[:, 0:3] > range1) * (pts_all[:, 0:3] < range2)
        idx = idx[:, 0] * idx[:, 1] * idx[:, 2]

        # filter
        pts_all_bridge = pts_all[idx]
        pts_all_non_bridge = pts_all[np.invert(idx)]

        # save points in bbox as *.obj file
        # from blender(normal) 3D space to *.obj 3D space, important!!
        pts_all_bridge[:, [1, 2]] = pts_all_bridge[:, [2, 1]]
        pts_all_non_bridge[:, [1, 2]] = pts_all_non_bridge[:, [2, 1]]
        fn_write = 'result/seg_%03d.obj' % i
        f = open(fn_write, 'w')
        for point in pts_all_bridge:
            f.write('v %f %f %f %f %f %f\n' % tuple(np.concatenate((point[0:3], (1, 0, 0)))))  # write xyz and red color
        for point in pts_all_non_bridge:
            f.write('v %f %f %f %f %f %f\n' % tuple(np.concatenate((point[0:3], (0.75, 0.75, 0.75)))))  # write xyz and gray color
        f.close()

    else:
        break


