# 差分进化算法求解Rosenbrock；DE: Differential Evolution
import numpy as np
import matplotlib.pyplot as plt
from HeuristicAlgorithms import DifferentialEvolution as DE

eps = 1e-16  # 收敛精度
pop_size = 20  # 种群大小
dimensions = 2  # 维度
x_min = np.array([-2, -2])  # 变量下界
x_max = np.array([2, 2])  # 变量上界
max_iter = 200  # 最大迭代次数
scaling_factor = 0.8  # 缩放因子，通常在0~2.0之间
crossover_rate = 0.8  # 交叉概率
strategy = 2  # 交叉选择策略，此处为1
pop = np.zeros((pop_size, dimensions))  # 初始化种群
for i in range(pop_size):
    pop[i, :] = np.random.rand(2) * (x_max - x_min) + x_min
pop_old = np.zeros(pop.shape)  # 保存当前种群
target_val = np.zeros(pop_size)  # 保存当前种群目标函数值
ibest = 0  # 保存当前最优个体索引----第一个
target_val[0] = DE.rosenbrock(pop[ibest, :])  # 计算当前种群目标函数值---第一个
best_val = 1e6  # 保存当前最优目标函数值
for i in range(1, pop_size):
    target_val[i] = DE.rosenbrock(pop[i, :])
    if target_val[i] < best_val:  # 更新最优个体
        best_val = target_val[i]
        ibest = i
best_x = pop[ibest, :]  # 保存当前最优个体
best_v = best_val  # 保存当前最优目标函数值
# 设计支持5种差分更新策略：从种群中随机选出5个个体参与更新
pm1 = np.zeros((pop_size, dimensions))
pm2 = np.zeros((pop_size, dimensions))
pm3 = np.zeros((pop_size, dimensions))
pm4 = np.zeros((pop_size, dimensions))
pm5 = np.zeros((pop_size, dimensions))
best_m = np.zeros((pop_size, dimensions))  # 保存当前种群最优个体
fresh_m = np.zeros((pop_size, dimensions))  # 更新后的个体
rotate_idx = np.arange(0, pop_size, 1)  # 旋转索引数组
rotate_dim = np.arange(0, dimensions, 1)  # 旋转维度数组
rotate_rt = np.zeros(pop_size)  # 另一个旋转索引数组大小
cross_rtd = np.zeros(dimensions)  # 另一个旋转维度数组大小----指数交叉的维度
# 迭代计算
iter = 0
while iter < max_iter and best_val > eps:  # 迭代计算, 当小于最大迭代或精度未达到要求
    # 增加多样性对原种群进行混合旋转操作
    pop_old = pop.copy()  # 保留原种群
    mix_ind = np.random.choice(4, 4, replace=False)  # 随机选择4个个体进行混合旋转
    pop_ind1 = np.random.choice(pop_size, pop_size, replace=False)  # 随机选择20个个体---种群1索引随机重排
    rt = np.mod(rotate_idx + mix_ind[0], pop_size)  # 旋转索引2---求余操作
    pop_ind2 = pop_ind1[rt]  # 种群2索引随机重排---由种群1旋转得到
    rt = np.mod(rotate_idx + mix_ind[1], pop_size)  # 旋转索引3---求余操作
    pop_ind3 = pop_ind2[rt]  # 种群3索引随机重排---由种群2旋转得到
    rt = np.mod(rotate_idx + mix_ind[2], pop_size)  # 旋转索引4---求余操作
    pop_ind4 = pop_ind3[rt]  # 种群4索引随机重排---由种群3旋转得到
    rt = np.mod(rotate_idx + mix_ind[3], pop_size)  # 旋转索引5---求余操作
    pop_ind5 = pop_ind4[rt]  # 种群5索引随机重排---由种群4旋转得到
    for i in range(pop_size):
        best_m[i, :] = best_x.copy()  # 保存当前种群最优个体
    # 混合旋转后的种群
    pm1=pop_old[pop_ind1, :]        # 混合旋转后种群1
    pm2=pop_old[pop_ind2, :]        # 混合旋转后种群2
    pm3=pop_old[pop_ind3, :]        # 混合旋转后种群3
    pm4=pop_old[pop_ind4, :]        # 混合旋转后种群4
    pm5=pop_old[pop_ind5, :]        # 混合旋转后种群5
    # 随机产生屏蔽字
    mui = (np.random.rand(pop_size, dimensions) < crossover_rate).astype(int)  # 产生随机数，小于交叉概率的为1，大于交叉概率的为0
    mui = np.sort(mui.T, axis=0)  # 按列排序
    for i in range(pop_size):
        n = np.floor(dimensions * np.random.rand(1)).astype(np.int16)  # 随机选择一个整数
        rtd = np.mod(rotate_dim + n, dimensions)  # 旋转维度---求余操作
        mui[:, i] = mui[rtd, i]  # 旋转维度---交换
    mui = mui.T  # 转置
    mpo = (mui < 0.5).astype(int)  # 产生屏蔽字----全部翻转
    # 交叉变异
    if strategy == 1:  # 简单差分策略
        fresh_m = pm1 + scaling_factor * (pm2 - pm3)  # 差分更新
        fresh_m = pop_old * mpo + fresh_m * mui  # 变异---二项式交叉
    elif strategy == 2:  # 基因重组差分策略
        fresh_m = best_m + scaling_factor * (pm1 - pm2)  # 差分更新
        fresh_m = pop_old * mpo + fresh_m * mui  # 变异---二项式交叉
    elif strategy == 3:  # 当前到目标差分策略
        fresh_m = pm3 + scaling_factor * (best_m - pop_old) + scaling_factor * (pm1 - pm2)  # 差分更新
        fresh_m = pop_old * mpo + fresh_m * mui  # 变异---二项式交叉
    elif strategy == 4:  # 当前到目标差分策略
        fresh_m = best_m + scaling_factor * (pm1 - pm2 + pm3 - pm4)  # 差分更新
        fresh_m = pop_old * mpo + fresh_m * mui  # 变异---二项式交叉
    elif strategy == 5:  # 当前到目标差分策略
        fresh_m = pm1 + scaling_factor * (pm2 - pm3 + pm4 - pm5)  # 差分更新
        fresh_m = pop_old * mpo + fresh_m * mui  # 变异---二项式交叉
    # 交叉变异后，对超出边界的个体进行裁剪---选择---
    for i in range(pop_size):
        temp_val = DE.rosenbrock(fresh_m[i, :])
        if temp_val < target_val[i]:  # 更新个体
            pop[i, :] = fresh_m[i, :]
            target_val[i] = temp_val
            if temp_val < best_val:  # 更新最优个体
                best_val = temp_val
                best_x = pop[i, :]
                ibest = i
    print('iter:', iter, 'best_val:', best_val, 'best_x:', best_x)
    iter += 1
# 输出结果
print('best_val:', best_val, 'best_x:', best_x)
