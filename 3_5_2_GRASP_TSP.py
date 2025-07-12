# 贪心随机自适应搜索算法（GRASP）解决旅行商问题（TSP）
# GRASP: Greedy Randomized Adaptive Search Procedure
# TSP: Traveling Salesman Problem
import numpy as np
from HeuristicAlgorithms import LocalSearch as LS

#  初始化
ls = LS()
prob_size = ls.gr17_matrix.shape[0]  # 城市数量
iterations = 1000  # 迭代次数
alpha = 0.5  # 随机选择概率，较小偏向贪心，较大偏向随机
count = 0  # 记录迭代次数
best_solution = np.zeros(prob_size + 1)  # 记录最优解
best_value = 1e9  # 记录最优解的长度
repeat_local = 5  # 局部搜索次数
repeat_RCL = 5  # RCL选择次数 RCL: Restricted Candidate List 受限候选者列表
while count < iterations:
    for i in range(repeat_RCL):  # RCL选择
        [rcl_value, rcl_solution] = ls.grasp_rcl(ls.gr17_matrix, alpha, prob_size)  # 贪心随机启发式搜索
        if rcl_value < best_value:
            best_value = rcl_value
            best_solution = rcl_solution
    for i in range(repeat_local):  # 局部搜索
        [local_value, local_solution] = ls.grasp_local_search(ls.gr17_matrix, best_solution)
        if local_value < best_value:
            best_value = local_value
            best_solution = local_solution
    count += 1
    print('迭代次数：', count, '最优解长度：', best_value, '最优解：', best_solution)
print('最优解长度为：', best_value)
print('最优解为：', best_solution)
