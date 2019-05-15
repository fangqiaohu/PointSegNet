import numpy as np
import glob
import os
import pandas as pd
from pyquaternion import Quaternion


root_dir = './data/pts_no_rot/'
fn_list = glob.glob(root_dir + '*.csv')
fn_list.sort()
ids = [os.path.basename(fn).split('.')[0] for fn in fn_list]


for id in ids:

    pts_name = os.path.join(root_dir, id + '.csv')
    pts_bbox_name = os.path.join(root_dir, id + '.npy')

    pts_all = np.array(pd.read_csv(pts_name))[:, 1:11]
    pts_bbox = np.load(pts_bbox_name)

    # un-rotate
    pts_all[:, 0:3] = np.dot(pts_all[:, 0:3], Quaternion(pts_bbox[6:10]).rotation_matrix)

    # write *.csv
    fn_write = pts_name
    df = pd.DataFrame(pts_all, columns=['x', 'y', 'z', 'nx', 'ny', 'nz', 'r', 'g', 'b', 'label'])
    df.to_csv(fn_write)

    # # check label
    # # filter
    range1 = pts_bbox[0:3] - pts_bbox[3:6] / 2
    range2 = pts_bbox[0:3] + pts_bbox[3:6] / 2
    idx = (pts_all[:, 0:3] > range1) * (pts_all[:, 0:3] < range2)
    idx = idx[:, 0] * idx[:, 1] * idx[:, 2]
    pts_all = pts_all[idx]
    print('points after bbox:', pts_all.shape)

    # save points in bbox as *.obj file
    # from blender(normal) 3D space to *.obj 3D space, important!!
    pts_all[:, [1, 2]] = pts_all[:, [2, 1]]
    fn_write = pts_name.replace('.csv', '.obj')
    f = open(fn_write, 'w')
    for point in pts_all:
        f.write('v %f %f %f %f %f %f\n' % tuple(np.concatenate((point[0:3], point[6:9]))))  # write xyz and rgb
        # f.write('v %f %f %f\n' % tuple(point[0:3]))  # write only xyz
    f.close()
