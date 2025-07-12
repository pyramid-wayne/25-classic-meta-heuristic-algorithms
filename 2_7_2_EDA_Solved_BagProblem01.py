# 分布估计算法,EDA:estimate distribution algorithm，解决背包问题

import numpy as np
from HeuristicAlgorithms import EstimationOfDistributionAlgorithm as EDA

# 背包问题
EDA = EDA()
iteration = 1000  # 迭代次数
popsize = 200  # 种群大小
weightMax = 6404180  # 背包最大承重
learningRate = 0.3  # 学习率
maxSpec = 200  # 最大变异刷新次数
dominantNum = 5  # 适应度前几名
dim = EDA.weight.shape[0]  # 变量维度
print("变量维度：", dim)
# 初始化种群
prob = 0.5 * np.ones(dim)  # 初始化概率
best_solution = np.zeros([iteration, dim + 1])  # 存储每次迭代的最优解
species = np.zeros([popsize, dim])  # 存储种群---保存各个体受概率分布影响产生的解
for I in range(iteration):
    flag = 0
    i = 0
    while i < popsize:  # 针对每个个体
        rand_num = np.random.rand(dim)  # 生成0-1之间的随机数,大小为种群数量
        species[i, :] = (rand_num < prob).astype(int)  # 根据概率分布生成个体----0,1保存
        weightSum = np.sum(species[i, :] * EDA.weight)  # 计算个体重量
        if flag >= maxSpec:  # 如果变异刷新次数达到最大值，则重新生成个体
            species[i, :] = np.zeros(dim)  # 多次仍难达到要求，重新生成个体
            flag = 0
        elif weightSum > weightMax:  # 如果个体重量超过背包最大承重，则重新生成个体
            i = i - 1
            flag += 1
        else:
            flag = 0
        i += 1
    # 计算适应度
    fitness = np.zeros(popsize)
    for i in range(popsize):
        fitness[i] = np.sum(species[i, :] * EDA.value)  # 计算个体价值---适应度
    # 选择适应度前几名
    [fitness, sort_index] = EDA.sort_arr_indices(fitness, descending=True)  # 降序排列
    best_solution[I, 0] = fitness[0]  # 存储每次迭代的最优解
    # best_solution[I, 1] = fitness[0]  # 存储每次迭代的最优解
    for i in range(1, dim + 1):
        best_solution[I, i] = species[sort_index[0], i - 1]
    # 选取种群中的优势群体
    domSpec = np.zeros([dominantNum, dim])  # 存储优势群体<popsize
    for i in range(dominantNum):
        domSpec[i, :] = species[sort_index[i], :]  # 选取优势群体
    # 更新概率分布
    ones_num = np.sum(domSpec, axis=0)  # 统计优势群体中1的个数
    prob = (1 - learningRate) * prob + learningRate * ones_num / dominantNum
print("第", iteration-1, "次迭代，最优解为：", best_solution[iteration-1, 0], " 最优组合为：", best_solution[iteration-1, 1:])
print('sum_weight: ',np.sum(best_solution[iteration-1, 1:]*EDA.weight))
print('sum_value: ',np.sum(best_solution[iteration-1, 1:]*EDA.value))
