from __future__ import print_function
import os
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset
import mlt_dataprocess
import mlt_model
import argparse
import mlt_train0
import mlt_train1
import mlt_train1_test2
import mlt_train2
import sys

# sys.path.append("./")
work_dir = os.path.dirname(os.path.abspath(__file__))
Data = os.path.join(work_dir,'data')
N_LOS_switch = False  # 这里什么意思
state = 0
torch.cuda.set_device(0)
parser = argparse.ArgumentParser(description='Mltask prediction & classification')  # 解析器
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--datapath', type=str, default='data/npy/all_all_train.npy')
parser.add_argument('--datasplit_or_not', type=bool, default=0)
parser.add_argument('--traindata', type=str, default=os.path.join(Data, "without_SH.npy"))
parser.add_argument('--testdata', type=str, default=os.path.join(Data, "only_SH.npy"))
parser.add_argument('--period', type=int, default=1)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--step_size', type=int, default=30)
parser.add_argument('--aug_switch', type=bool, default=True)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--maskvalue', type=float, default=1e-5)
parser.add_argument('--epochs_P1', type=int, default=20)
parser.add_argument('--epochs_P2', type=int, default=20)

# **************************修改关键参数******************************
# keyinfo = 'Task10 MTL All-All-2-noite-nores-noaug'
keyinfo = 'Based_on_the_best_netTheia2&netLos_changed'  # 这里什么鬼意思?
characteristic_index = np.array([0, 1, 2, 3, 4, 5, 6])
target_index = np.array([1, 2, 3, 4, 5, 6])

# channel characteristics : height K phi theta p t los
if len([i for i in range(len(target_index)) if target_index[i] == 6]) > 0:  # 这一小快是判断target_index里面是否有N_LOS
    N_LOS_switch = True
    state = 1
    if len(target_index) == 1:  # 有N_los但是同时target中只有N_Los
        state = 2
parser.add_argument('--characteristic_index', type=np.ndarray, default=characteristic_index)
parser.add_argument('--target_index', type=np.ndarray, default=target_index)
parser.add_argument('--N_LOS_switch', type=bool, default=N_LOS_switch)
parser.add_argument('--state', type=int, default=state)
parser.add_argument('--weight_lastname', type=str, default=keyinfo + '.pth')
parser.add_argument('--logname', type=str, default=keyinfo + '.txt')
parser.add_argument('--jpgname', type=str, default=keyinfo + '.jpg')
parser.add_argument('--csvname', type=str, default=keyinfo + '.csv')
parser.add_argument('--argsname', type=str, default='args' + keyinfo + '.txt')
parser.add_argument('--wight_lastname', type=str, default=keyinfo + '.pth')
args = parser.parse_args()

# random seed set
mlt_dataprocess.seed_everything(args)

# 第一次数据转化时需要
# vit_dataprocess.mat2npy()     这里数据我还是不会处理，感觉提麻烦的

mlt_dataprocess.mat2npy_train()
# mlt_dataprocess.mat2npy_test()


