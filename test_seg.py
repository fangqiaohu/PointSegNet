import os
import glob
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_loader import MyDataset


# hyper-parameters
batch_size = 1
USE_CUDA = torch.cuda.is_available()


root_dir = './data/pts_no_rot/'
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
model = torch.load('model/model_000.pkl', map_location='cpu').eval()

if USE_CUDA:
    model = model.cuda()

for i, data in enumerate(test_loader, 0):

    if i in [0, 1, 2]:

        pts_xyz = data['pts_xyz']
        pts_label = data['pts_label']

        pts_xyz = pts_xyz.float()
        pts_label = pts_label.float()

        if USE_CUDA:
            pts_xyz = pts_xyz.cuda()
            pts_label = pts_label.cuda()

        pts_pred = model(pts_xyz)
        pts_pred = F.softmax(pts_pred, dim=1)

        pts_pred = np.array(pts_pred.detach().squeeze())
        pts_label = np.array(pts_label.detach().squeeze(), dtype=int)

        pts_pred_hard = pts_pred[1, :] > 0.2  # threshold
        # pts_pred_hard = pts_pred[1, :] > pts_pred[0, :]  # compare
        # pts_pred_hard = pts_pred[1, :] - 0.5 > pts_pred[0, :]  # un-balanced compare
        print(pts_pred)
        print(np.argmax(pts_pred, axis=0))
        print(pts_label)
        pts_all = data['pts_all']
        pts_all = np.array(pts_all.detach().squeeze())

        # filter
        pts_all_bridge = pts_all[pts_pred_hard]
        pts_all_non_bridge = pts_all[np.invert(pts_pred_hard)]

        # save points in bbox as *.obj file
        # from blender(normal) 3D space to *.obj 3D space, important!!
        pts_all_bridge[:, [1, 2]] = pts_all_bridge[:, [2, 1]]
        pts_all_non_bridge[:, [1, 2]] = pts_all_non_bridge[:, [2, 1]]
        fn_write = 'result/seg_%03d.obj' % i
        f = open(fn_write, 'w')
        for point in pts_all_bridge:
            f.write('v %f %f %f %f %f %f\n' % tuple(np.concatenate((point[0:3], (1, 0, 0)))))  # write xyz and red
        for point in pts_all_non_bridge:
            f.write('v %f %f %f %f %f %f\n' % tuple(np.concatenate((point[0:3], (0.75, 0.75, 0.75)))))  # write xyz and gray
        f.close()

    else:
        break


