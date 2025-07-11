"""
进化规划算法求解二进制函数问题，EP:Evolutionary Programming; x1∈[-0.3,12.1],x2∈[4.1,5.8]
"""
import numpy as np
import matplotlib.pyplot as plt
from HeuristicAlgorithms import EvolutionaryProgramming as EP

# 初始化
EP = EP()
miu = 100  # 种群大小
max_iter = 1000  # 最大迭代次数
sigma_x1 = np.ones(miu)  # 初始种群x1分量的标准差初始值
sigma_x2 = np.ones(miu)
pop_x1 = np.zeros(miu)  # 存放每代个体x1分量的值
pop_x2 = np.zeros(miu)  # 存放每代个体x2分量的值
# 平均变异：pop
for i in range(5):
    for j in range(20):
        pop_x1[i * 20 + j] = -0.3 + (j - 1) * 0.62 + np.random.rand() * 0.62  # x1 分为20等份
        pop_x2[i * 20 + j] = 4.1 + (i - 1) * 0.34 + np.random.rand() * 0.34  # x2 分为5等份
# 计算初始种群适应度fitness
fit_pop = EP.compute_fitness(pop_x1, pop_x2)
[fit_output, sort_index] = EP.sort_arr_indices(fit_pop, descending=True)  # 降序# 排序
# 迭代
for i in range(max_iter):
    pop_compute_x1 = pop_x1 + np.sqrt(sigma_x1) * np.random.randn(miu)  # 变异个体x1分量
    for k in range(miu):
        while pop_compute_x1[k] < -0.3 or pop_compute_x1[k] > 12.1:  # 变异个体x1分量超出范围，重新生成
            pop_compute_x1[k] = pop_x1[k] + np.sqrt(sigma_x1[k]) * np.random.randn()
    pop_compute_x2 = pop_x2 + np.sqrt(sigma_x2) * np.random.randn(miu)  # 变异个体x2分量
    for k in range(miu):
        while pop_compute_x2[k] < 4.1 or pop_compute_x2[k] > 5.8:  # 变异个体x2分量超出范围，重新生成
            pop_compute_x2[k] = pop_x2[k] + np.sqrt(sigma_x2[k]) * np.random.randn()
    fit_compute = EP.compute_fitness(pop_compute_x1, pop_compute_x2)  # 计算变异个体适应度
    sigma_x1 = sigma_x1 + np.sqrt(sigma_x1) * np.random.randn(miu)  # 更新标准差
    sigma_x1 = np.abs(sigma_x1)  # 标准差取正值
    sigma_x2 = sigma_x2 + np.sqrt(sigma_x2) * np.random.randn(miu)  # 更新标准差
    sigma_x2 = np.abs(sigma_x2)  # 标准差取正值
    # 采用q竞争法选择个体组成新种群---合并父代和变异个体
    pop_temp_x1 = np.concatenate((pop_x1, pop_compute_x1), axis=0)
    pop_temp_x2 = np.concatenate((pop_x2, pop_compute_x2), axis=0)
    fit_temp = EP.compute_fitness(pop_temp_x1, pop_temp_x2)  # 计算父代和变异适应度
    q_score = np.zeros(miu * 2)  # q竞争得分
    for k in range(miu * 2):
        # 从0到miu-1中随机选择90个不重复的数作为q竞争法选择的裁判位置
        position = np.random.choice(miu, 90, replace=False)
        judge_x1 = pop_temp_x1[position]  # 裁判个体x1分量
        judge_x2 = pop_temp_x2[position]  # 裁判个体x2分量
        fit_judge = EP.compute_fitness(judge_x1, judge_x2)  # 裁判适应度
        for m in range(90):
            if fit_temp[k] > fit_judge[m]:
                q_score[k] = q_score[k] + 1
    # 根据q_score排序
    [q_score_output, q_score_index] = EP.sort_arr_indices(q_score, descending=True)
    # 根据q_score_index选择新种群
    pop_x1 = pop_temp_x1[q_score_index[0:miu]]
    pop_x2 = pop_temp_x2[q_score_index[0:miu]]
    fit_pop = EP.compute_fitness(pop_x1, pop_x2)  # 计算新种群适应度
    # 排序
    [fit_output, sort_index] = EP.sort_arr_indices(fit_pop, descending=True)  # 降序
    print("第{}次迭代".format(i), "最优适应度：", fit_output[0], "最优解：", pop_x1[sort_index[0]], pop_x2[sort_index[0]])
