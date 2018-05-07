# -*- coding: utf-8 -*-
# 格子玻尔兹曼D2Q9模型DEMO程序
from settings import *
import tools
import scipy.io as sio
import time

# 开始计时
time_start = time.time()


def set_up():

    # 读取流场信息
    ff = tools.get_fluidfield_info_2D(ff_name = 'TEST.png')

    # 读取输入文件并创建保存路径
    par_TPFD, save_path = tools.get_inputfile_and_make_saveworkshop()

    # 初始化计算环境并保存流场信息
    omega_f, r, Fpr, rho_lu, Tr, par_spare_ff = init_D2Q9(ff, par_TPFD, save_path)

    # 计算主体
    main_loop(save_path, omega_f, r, Fpr, rho_lu, Tr, par_spare_ff, ff)

    # 计算完成提示+
    total_time = time.time() - time_start
    print('\nCalculation finished, totally takes %s.Thanks for use!'
          % time.strftime('%Hh%Mm%Ss', time.gmtime(total_time)))


def init_D2Q9(ff, par_TPFD, save_path):

    """初始化程序，读取流场信息"""

    # 计算格子温度、压力、压差等信息
    Tr, pr, Fpr = tools.get_lu_tem_pre_dp(par_TPFD=par_TPFD)


    # 计算给定温压条件下的流体密度
    rho_real, rho_lu = tools.get_rho_real_and_rho_lu(Tr, pr)

    # 转化孔径单位
    L_real = 10 ** (-9) * par_TPFD['r'][0]  # 单位：米（m）

    # 稀疏化流场提升运算效率
    par_spare_ff = tools.sparse_array_D2Q9(ff)

    # 计算松弛时间、克努森数、粘度等参数信息
    kn, kne, omega_f, nu_f, nu_fe, r = tools.get_kn_omega_r(rho_real, L_real, ff.shape[1])
    sio.savemat(save_path + fslash + 'kne_kn', {'kn': kn, 'kne': kne, 'miu': nu_f, 'miue': nu_fe})

    return omega_f, r, Fpr, rho_lu, Tr, par_spare_ff


def main_simulation(fIn, omega, r, par_spare_ff, Fpx, Fpy, Tr, flag):

    t_begin = time.time()
    cur_iter = 0  # 初始化当前循环次数
    rho = np.sum(fIn, axis=0)
    lr_f = 0 * rho
    i = 0
    ux_macro_avg = {}
    uy_macro_avg = {}
    stopflag = 0
    while 1:
        cur_iter += 1
        if cur_iter % 1000 == 999:
            lr_f = 1 * rho

        # 计算宏观物理量
        ux, uy, delta_ux, delta_uy, ux_macro, uy_macro, rho, p = \
            tools.get_macro_phy_quan_D2Q9(fIn, par_spare_ff[0], par_spare_ff[1], Fpx, Fpy, Tr, 0)

        # 程序终止判定

        if cur_iter % 1000 == 0:
            n_toler = np.sum((rho - lr_f) ** 2) / par_spare_ff[4]
            print('> Loop times: %07d, current variance: %.2e.' % (cur_iter, n_toler))
            if n_toler < Toler and cur_iter >= Min_Iter:
                stopflag = 1

        if cur_iter % Max_Iter == 0:
            stopflag = 1

        if stopflag == 1:
            ux_macro_avg[i] = ux_macro
            uy_macro_avg[i] = uy_macro
            i = i + 1
            if i > 5:
                ux_macr = (ux_macro_avg[0] + ux_macro_avg[1] + ux_macro_avg[2] + ux_macro_avg[3] + ux_macro_avg[4] +
                           ux_macro_avg[5]) / 6
                uy_macr = (uy_macro_avg[0] + uy_macro_avg[1] + uy_macro_avg[2] + uy_macro_avg[3] + uy_macro_avg[4] +
                           uy_macro_avg[5]) / 6
                print('The ' + str(flag) + 'th model\'s calculation finished...')
                t = time.time() - t_begin
                return t, ux_macr, uy_macr, rho, p

        # 碰撞
        fOut = tools.collision_D2Q9(fIn, ux, uy, delta_ux, delta_uy, rho, omega)

        # 迁移
        fIn = tools.migration_D2Q9(fOut, par_spare_ff[2], par_spare_ff[3], r)


def main_loop(save_path, omega, r, Fpr, rho_lu, Tr, par_spare_ff, ff):
    """主循环"""
    flag = 0
    print('Loop begins...')
    for omega_i, r_i, Fpr_i, rho_lu_i, Tr_i in zip(omega, r, Fpr, rho_lu, Tr):
        flag = flag + 1
        print('Current model is No.' + str(flag) + '...')

        ux_matrix = 1 * ff
        uy_matrix = 1 * ff
        rho_matrix = 1 * ff
        p_matrix = 1 * ff

        # 压力梯度
        Fpx = Fpr_i * np.ones(par_spare_ff[4])
        Fpy = 0 * np.ones(par_spare_ff[4])

        # 初始化密度分布函数
        fIn = tools.init_fIn_D2Q9(rho_lu_i, par_spare_ff[4])

        fname = save_path + fslash + str(flag)

        # 执行碰撞迁移过程
        t, ux_macr, uy_macr, rho, p = \
            main_simulation(fIn, omega_i, r_i, par_spare_ff, Fpx, Fpy, Tr_i, flag)

        # 还原流场
        ux_matrix[ux_matrix == 1] = ux_macr
        uy_matrix[uy_matrix == 1] = uy_macr
        rho_matrix[rho_matrix == 1] = rho
        p_matrix[p_matrix == 1] = p

        # 保存计算结果，并提示
        sio.savemat(fname, {'ux': ux_matrix, 'uy': uy_matrix, 'rho': rho_matrix, 'pre': p_matrix})
        print('This model takes about %s'% time.strftime('%Hh%Mm%Ss', time.gmtime(t)))


set_up()








