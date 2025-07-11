"""
进化策略ES求解二元函数,ES: Evolution Strategy;    二元函数: y=21.5+x1*sin(4*pi*x1)+x2*sin(2*pi*x2)
种群规模：40；迭代次数：200；x1∈[-3.0,12.1],x2∈[4.1,5.8]
"""
import numpy as np
from HeuristicAlgorithms import EvolutionStrategy as ES
import matplotlib
from matplotlib import pyplot as plt

"设置字体"
matplotlib.rc("font", family='Microsoft YaHei')


"初始化参数"
miu = 40  # 种群规模---随机生成μ个个体
x1 = 15.1 * np.random.rand(1, miu) - 3.0  # 初始解：x1∈[-3.0,12.1]
x2 = 1.7 * np.random.rand(1, miu) + 4.1  # 初始解：x2∈[4.1,5.8]
X = np.concatenate((x1, x2), axis=0)  # 初始种群
sigma = np.random.rand(2, miu)  # 初始标准差
MaxIter = 600  # 迭代次数
maxy = 0.0  # 初始最优值---最大值
global_best_x = []  # 存储最优解
max_y_list = []  # 存储每次迭代的最优值
mean_y_list = []  # 存储每次迭代的平均值
for iter in range(MaxIter):
    lamda = 1  # 生成λ个变异个体
    offspring = []
    while lamda < 7 * miu:  # 经验值：λ/μ=7
        pos = np.uint16(np.random.rand(1, 2) * (miu - 1))  # 随机选择两个个体，0~miu-1之间
        pa1 = X[:, pos[0, 0]]  # 选择个体1      提取两个位置的(X,σ)
        pa2 = X[:, pos[0, 1]]  # 选择个体2
        # 生成变异个体----X采用离散重组
        option = np.zeros([2, 1])
        if np.random.rand() < 0.5:  # 0.5的概率进行离散重组
            option[0, 0] = pa1[0]  # 0.5的概率选择个体1的X
        else:
            option[0, 0] = pa2[0]
        if np.random.rand() < 0.5:  # 0.5的概率选择个体2的X
            option[1, 0] = pa1[1]
        else:
            option[1, 0] = pa2[1]
        # 生成变异个体----σ采用中值重组
        sigma1 = 0.5 * (sigma[:, pos[0, 0]] + sigma[:, pos[0, 1]])
        sigma1 = sigma1.reshape(-1, 1)
        Y = option + sigma1 * np.random.randn(2, 1)  # 生成变异个体
        if -3 <= Y[0, 0] <= 12.1 and 4.1 <= Y[1, 0] <= 5.8:  # 判断变异个体是否在定义域内
            # offspring=np.concatenate((offspring, sigma1), axis=1)  # 将变异个体和对应的σ拼接
            offspring.append(Y)  # 将变异个体和对应的σ拼接，在可行域内，则将更新后的个体加入子代群体
            lamda += 1  # 计数
        # break
    new_solved = np.squeeze(offspring).T  # 采用μ、λ策略，得到λ个新解 np.array(offspring)

    # 计算目标解
    binary_solved = []
    for i in range(new_solved.shape[1]):
        par_x1 = new_solved[0, i]
        par_x2 = new_solved[1, i]
        binary_solved.append(ES.binaryfunc(par_x1, par_x2))
    # 从λ个后代，选出μ个最好解
    [new_sort_binary, sort_idx_mat] = ES.sort_list_indices(binary_solved)
    I1 = sort_idx_mat[-miu:]  # 取出最后μ个索引
    best_solved = new_solved[:, I1]  # 取出最后μ个个体
    # 更新种群
    if new_sort_binary[-1] > maxy:  # 判断是否更新最优解
        maxy = new_sort_binary[-1]  # 更新最优解
        global_best_x = best_solved[:, -1]
    max_y_list.append(maxy)
    mean_y_list.append(np.mean(new_sort_binary))
print('最优解：', global_best_x)
print('最优值：', maxy)
# 绘制结果
plt.plot(max_y_list, label='max_y')
plt.plot(mean_y_list, label='mean_y')
plt.title("迭代计算")   # 标题, fontproperties='SimHei'
plt.xlabel("迭代次数")        # X轴标签, fontproperties='SimHei'
plt.ylabel("迭代最大值")        # Y轴标签, fontproperties='SimHei'
plt.grid(True)           # 网格
plt.show()               # 显示图形

