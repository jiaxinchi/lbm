# -*- coding: utf-8 -*-
import numpy as np

# ------------------------------------不可变量--------------------------------------------------

PAR_D2Q9 = {
    'cs': 1. / np.sqrt(3),
    't': np.array([4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 36, 1. / 36, 1. / 36, 1. / 36]),
    'cx': np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]),
    'cy': np.array([0, 0, 1, 0, -1, 1, 1, -1, -1]),
    'op': np.array([0, 3, 4, 1, 2, 7, 8, 5, 6]), # 反弹格式
    'mb': np.array([0, 3, 4, 1, 2, 8, 7, 6, 5])  # 镜面反弹格式
}

PAR_FLUID = {
    """流体(甲烷)的属性参数"""
    'Tc_lu': 1,
    'Tc': 190.55,
    'rhoc_lu': 1,
    'rhoc': 125.6896,
    'pc_lu': 1,
    'pc': 4.59 * 10 ** 6,
    'd': 0.38 * 10 ** (-9),
    'm': 2.658 * 10 ** (-26),
    'b': 2 * np.pi * 0.38 * 10 ** (-9) ** 3 / (3 * 2.658 * 10 ** (-26))
    }

# 运行环境选择
while 1:
    k = input("请选择运行环境（1: windows 2: linux）：")
    if k == '2':
        fslash = '//'
        break
    elif k == '1':
        fslash = '\\'
        break
    else:
        print("WRONG ORDER，TRY AGAIN...")

# ----------------------------------------可变量-----------------------------------------------

Toler = 1.0e-15  # 计算精度
Max_Iter = 50000  # 最大循环次数
Min_Iter = 10000  # 最小循环次数
Gc = - 0.4 # 流体之间相互作用力强度
