# -*- coding: utf-8 -*-
from settings import *
from functools import wraps
import matplotlib.image as mplimg  # 读取图片
import scipy.io as sio
import os
import random
import sympy
import math
import time


def metric(text="this func"):

    """装饰器：统计函数运行时间"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            time_now = time.time()
            f = func(*args,**kwargs)
            print((text + ' spent about %s')% time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - time_now)))
            return f
        return wrapper
    return decorator



def get_newest_file(file_list):

    """对给定的文件夹内文件排序，获取最后修改的文件"""

    file_list.sort(key=lambda f_list: os.path.getmtime(f_list))
    return file_list[-1]


def get_mat_file(path):

    """获取给定路径文件夹内的.mat文件"""

    f_list = os.listdir(path)
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '.mat':
            return i, os.path.splitext(i)[0]


def get_inputfile_and_make_saveworkshop():

    """读取输入文件"""

    dir_pro = os.getcwd()  # 获取程序工作目录
    dir_data = dir_pro + fslash + 'Data'
    tname, name = get_mat_file(dir_data)
    save_path = dir_pro + fslash + 'Data' + fslash + name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    input_par = sio.loadmat(dir_data + fslash + tname)
    return input_par, save_path


def get_lu_tem_pre_dp(par_TPFD):

    """获取格子温度、压力、压差等参数信息"""

    # 无量纲温度、压力、压差
    Tr = par_TPFD['T'][0] / PAR_FLUID['Tc']
    pr = 10 ** 6 * par_TPFD['p'][0] / PAR_FLUID['pc']
    Fpr = par_TPFD['Fp'][0] * 1

    return Tr, pr, Fpr


def circleshift(array, drx, dry):

    """矩阵循环平移"""

    array_cs = np.roll(np.roll(array, drx, axis=0), dry, axis=1)
    return array_cs


def feq_d2q9(rho, ux, uy, dr):

    """D2Q9模型平衡态分布函数"""

    u = PAR_D2Q9['cx'][dr] * ux + PAR_D2Q9['cy'][dr] * uy
    inner_pro_u = ux * ux + uy * uy
    feq = rho * PAR_D2Q9['t'][dr] * (1 + 3.0 * u + 4.5 * u * u - 1.5 * inner_pro_u)
    return feq


def rho2squ_fei_vdw(rho, Tr, flag):

    """气体状态方程，密度转换为有效密度的平方"""
    if flag == 0:
        # 范德华方程（对应态）
        squ_fei = 2.0*rho*((8*Tr/(3-rho)-3*rho)/8/Tr-1/3.0)/6.0/Gc
    return squ_fei


def get_fluidfield_info_2D(ff_name='TEST.png'):

    """读取流场信息"""

    ff_rgba = mplimg.imread(ff_name)
    ff = ff_rgba[:, :, 1]
    ff = np.transpose(ff, (1, 0))  # 转置
    return ff


def get_rho_real_and_rho_lu(Tr, pr):

    """通过Tr，pr获取真实密度和格子密度"""

    rho_real = 0 * Tr
    rho_lu = 0 * Tr
    rho = sympy.symbols('rho')
    ind = 0
    for T, p in zip(Tr, pr):
        rho_real[ind] = PAR_FLUID['rhoc'] * sympy.solve(8 * T * rho / (3 - rho) - 3 * rho ** 2 - p, rho)[0]
        ind = ind + 1
        rho_lu = rho_real / PAR_FLUID['rhoc'] * PAR_FLUID['rhoc_lu']  # 初始密度

    return rho_real, rho_lu


def get_kn_omega_r(rho_real, L_real, ly):

    """计算流场特征参数"""

    num = 0

    kn = np.zeros(len(rho_real))
    kne = np.zeros(len(rho_real))
    omega_f = np.zeros(len(rho_real))
    nu_f = np.zeros(len(rho_real))
    nu_fe = np.zeros(len(rho_real))
    r = np.zeros(len(rho_real))

    for rhoreal, Lreal in zip(rho_real, L_real):
        # 自由程
        lamb = PAR_FLUID['m'] / (np.sqrt(2) * np.pi * rhoreal * PAR_FLUID['d'] ** 2)

        # 理想和有效克努森数
        kn[num] = lamb / Lreal
        kne[num] = kn[num] * 2. / np.pi * math.atan(np.sqrt(2) * kn[num] ** (-3 / 4))

        # 径向分布函数
        x = 1 + 5. / 8 * PAR_FLUID['b'] * rhoreal + 0.2869 * (PAR_FLUID['b'] * rhoreal) ** 2 \
            + 0.1103 * (PAR_FLUID['b'] * rhoreal) ** 3 + 0.0386 * (PAR_FLUID['b'] * rhoreal) ** 4

        rhobx = rhoreal * PAR_FLUID['b'] * x

        # 流体的松弛参数
        taoe = np.sqrt(6 / np.pi) * (1 + 0.5 * rhobx) ** 2 * (ly-2) * kne[num] / x + 1 / 2
        tao = np.sqrt(6 / np.pi) * (1 + 0.5 * rhobx) ** 2 * (ly-2) * kn[num] / x + 1 / 2
        nu_fe[num] = (taoe - 0.5) * PAR_D2Q9['cs'] ** 2.  # 有效格子粘度系数
        nu_f[num] = (tao - 0.5) * PAR_D2Q9['cs'] ** 2.  # 理想格子粘度系数
        omega_f[num] = 1 / taoe

        # r 值的确定
        xigema = 1
        a1 = (2 - xigema) / xigema * (1 - 0.1817 * xigema)
        a2 = 1 / np.pi + 1 / 2 * a1 ** 2
        r[num] = 1 / (1 + np.sqrt(np.pi / 6) * ((1 / (ly-2)) ** 2 / (4 * kne[num])
                                                + a1 + (2 * a2 - 8 / np.pi) * kne[num]))
        num = num + 1

    return kn, kne, omega_f, nu_f, nu_fe, r


def sparse_array_D2Q9(ff):

    """稀疏化流场"""

    lx, ly = ff.shape
    num_fluid_lattice = int(np.sum(ff))
    # 孔隙度
    # porosity = num_fluid_lattice / ff.size

    # 孔壁（固体部分）
    # solid = np.ones((lx, ly), dtype=np.int) - ff

    # 稀疏化流场
    ff_sp = np.zeros((lx, ly), dtype=np.int)
    ff_sp[ff == 1] = np.array(range(1, num_fluid_lattice + 1))

    # ----------------------------------------------
    #     0 0 0 0 0 0 0     0 0 0 0 0 0 0
    #     0 1 1 1 0 0 0     0 1 2 3 0 0 0
    #     0 1 1 1 1 0 0  →  0 4 5 6 7 0 0
    #     0 0 1 1 0 0 0     0 0 0 8 9 0 0
    #     0 0 0 0 0 0 0     0 0 0 0 0 0 0
    # -----------------------------------------------

    solid_sp = np.zeros((lx, ly), dtype=np.int)
    solid_sp[ff == 0] = 1

    # 稀疏化周围格点
    ff_sp_aside = {}
    solid_sp_aside = {}
    for dr in range(9):

        ff_sp_aside_dr = circleshift(ff_sp, -PAR_D2Q9['cx'][dr], -PAR_D2Q9['cy'][dr])
        ff_sp_aside[dr] = ff_sp_aside_dr[ff_sp != 0]  # 相邻流体格点

        solid_sp_aside_dr = circleshift(solid_sp, -PAR_D2Q9['cx'][dr], -PAR_D2Q9['cy'][dr])
        solid_sp_aside[dr] = solid_sp_aside_dr[ff_sp != 0]  # 相邻格点固体密度

    # 迁移过程稀疏化
    fIn_id_op = {}
    fIn_id_mb = {}
    for dr in range(9):
        drop = PAR_D2Q9['op'][dr]
        drmb = PAR_D2Q9['mb'][dr]
        ff_sp_mig = circleshift(ff_sp, PAR_D2Q9['cx'][dr], PAR_D2Q9['cy'][dr])

        drffop = dr * np.ones((lx, ly), dtype=np.int)
        drffmb = dr * np.ones((lx, ly), dtype=np.int)

        # 与固体边界相邻的流体格点
        obst = ((ff_sp_mig == 0) == (ff_sp != 0))

        ff_sp_mig[obst] = ff_sp[obst]
        drffop[obst] = drop  # 反弹格式
        drffmb[obst] = drmb  # 镜面反弹
        fIn_id_op[dr] = np.array([ff_sp_mig[ff_sp != 0], drffop[ff_sp != 0]])  # 反弹格式迁移
        fIn_id_mb[dr] = np.array([ff_sp_mig[ff_sp != 0], drffmb[ff_sp != 0]])  # 镜面反弹迁移

    return [ff_sp_aside, solid_sp_aside, fIn_id_op, fIn_id_mb, num_fluid_lattice]


def init_fIn_D2Q9(rholu, num_fluid_lattice):

    """初始化速度分布函数配置"""
    fIn = np.zeros(shape=(9, num_fluid_lattice))
    r = []
    [r.append(np.array([random.random() for i in range(num_fluid_lattice)])) for dr in range(9)]
    drho = 0.0001
    for dr in range(9):
        fIn[dr] = PAR_D2Q9['t'][dr] * rholu * np.ones(num_fluid_lattice) * (1 + drho * (1 - 2 * r[dr]))
        # 流体的的密度
    return fIn


def get_macro_phy_quan_D2Q9(fIn, ff_sp_aside, solid_sp_aside, Fpx, Fpy, Tr, eosflag):

    """获取宏观物理量"""

    # 流体密度
    rho = np.sum(fIn, axis=0)
    rho_temp = np.append(rho, 0)

    ux = np.sum(PAR_D2Q9['cx'] * fIn.T, axis=1) / rho
    uy = np.sum(PAR_D2Q9['cy'] * fIn.T, axis=1) / rho

    rho_aside = np.array([rho_temp[ff_sp_aside[0] - 1],
                          rho_temp[ff_sp_aside[1] - 1], rho_temp[ff_sp_aside[2] - 1], rho_temp[ff_sp_aside[3] - 1],
                          rho_temp[ff_sp_aside[4] - 1],
                          rho_temp[ff_sp_aside[5] - 1], rho_temp[ff_sp_aside[6] - 1], rho_temp[ff_sp_aside[7] - 1],
                          rho_temp[ff_sp_aside[8] - 1]])

    fei2 = rho2squ_fei_vdw(rho, Tr, eosflag)
    fei = np.sqrt(fei2)
    fei2_aside = rho2squ_fei_vdw(rho_aside, Tr, eosflag)
    fei_aside = np.sqrt(fei2_aside)
    A = 0.2
    fc_fx = - A * Gc * (1 * (fei2_aside[1] - fei2_aside[3]) + 1. / 4 * (fei2_aside[5] - fei2_aside[7] + fei2_aside[8] - fei2_aside[6]))  + \
            (-(1 - A) * Gc * fei * (2 * (fei_aside[1] - fei_aside[3]) + 1. / 2 * (fei_aside[5] - fei_aside[7] + fei_aside[8] - fei_aside[6])))

    fc_fy = - A * Gc * (1 * (fei2_aside[2] - fei2_aside[4]) + 1. / 4 * (fei2_aside[5] - fei2_aside[7] + fei2_aside[6] - fei2_aside[8]))  + \
            (-(1 - A) * Gc * fei * (2 * (fei_aside[2] - fei_aside[4]) + 1. / 2 * (fei_aside[5] - fei_aside[7] + fei_aside[6] - fei_aside[8])))

    # solid_aside = np.array([solid_sp_aside[0],
    #                         solid_sp_aside[1], solid_sp_aside[2], solid_sp_aside[3], solid_sp_aside[4],
    #                         solid_sp_aside[5], solid_sp_aside[6], solid_sp_aside[7], solid_sp_aside[8]])

    fads_fx = 0
    fads_fy = 0

    delta_ux = (Fpx + fads_fx + fc_fx) / rho
    delta_uy = (Fpy + fads_fy + fc_fy) / rho

    ux_macro = ux + 0.5 * delta_ux
    uy_macro = uy + 0.5 * delta_uy

    p = 8 * rho * Tr / (3 - rho) - 3 * rho * rho
    return ux, uy, delta_ux, delta_uy, ux_macro, uy_macro, rho, p


def collision_D2Q9(fIn, ux, uy, delta_ux, delta_uy, rho, omega):

    """碰撞过程"""

    fOut = 1 * fIn
    for dr in range(9):
        # exact difference method(EDM)方法
        feq = feq_d2q9(rho, ux, uy, dr)
        delta_feq = feq_d2q9(rho, ux + delta_ux, uy + delta_uy, dr) - feq
        fOut[dr] = fIn[dr] - omega * (fIn[dr] - feq) + delta_feq
    return fOut


def migration_D2Q9(fOut, fIn_id_op, fIn_id_mb, r):

    """迁移过程"""

    fIn = 1 * fOut
    for dr in range(9):
        fIn[dr] = r * fOut[fIn_id_op[dr][1], fIn_id_op[dr][0] - 1] +\
                    (1 - r) * fOut[fIn_id_mb[dr][1], fIn_id_mb[dr][0] - 1]
    return fIn
