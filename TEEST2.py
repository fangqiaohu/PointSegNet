import numpy as np
import glob
import os
import pandas as pd
from pyquaternion import Quaternion

import torch
import torch.nn.functional as F

torch.manual_seed(1000)

a = torch.randn(24).view(3, 2, 4).float()

# print(a)
#
# a = a.transpose(2, 1).contiguous()
#
# print(a)
#
# a = a.view(-1, 2)
#
# print(a)

b = F.softmax(a, 0)

# print(b)
#
# b = b.view(3, 2, 4)

print(b)


