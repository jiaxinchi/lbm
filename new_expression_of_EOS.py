from settings import *
import tools
import scipy.io as sio
import time


@tools.metric(text='this model')
def set_up():

    ff = tools.get_fluidfield_info_2D(ff_name='EOS.png')
    omega = 0.5
    r = 1
    Fpr = 0
    rho_lu = 0.7
    Tr = 0.78
    par_spare_ff = tools.sparse_array_D2Q9(ff)

    ux_matrix = 1 * ff
    uy_matrix = 1 * ff
    rho_matrix = 1 * ff
    p_matrix = 1 * ff

    # 压力梯度
    Fpx = Fpr * np.ones(par_spare_ff[4])
    Fpy = 0 * np.ones(par_spare_ff[4])

    # 初始化密度分布函数
    fIn = tools.init_fIn_D2Q9(rho_lu, par_spare_ff[4])

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
                print('calculation finished...')
                break

        # 碰撞
        fOut = tools.collision_D2Q9(fIn, ux, uy, delta_ux, delta_uy, rho, omega)

        # 迁移
        fIn = tools.migration_D2Q9(fOut, par_spare_ff[2], par_spare_ff[3], r)

    ux_matrix[ux_matrix == 1] = ux_macr
    uy_matrix[uy_matrix == 1] = uy_macr
    rho_matrix[rho_matrix == 1] = rho
    p_matrix[p_matrix == 1] = p

    # 保存计算结果，并提示
    sio.savemat('EOS', {'ux': ux_matrix, 'uy': uy_matrix, 'rho': rho_matrix, 'pre': p_matrix})

set_up()