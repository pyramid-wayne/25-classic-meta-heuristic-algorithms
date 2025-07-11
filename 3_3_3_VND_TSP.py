# 变邻域搜索，VNS: Variable Neighborhood Search；解决TSP问题：旅行商问题
# 确定性变邻域下降算法，DVND: Deterministic Variable Neighborhood Descent

import numpy as np
from HeuristicAlgorithms import LocalSearch as LS

ls = LS()
iterMax = 5  # 最大迭代次数
iterx = 0  # 迭代次数
[row, col] = np.shape(ls.dist_matrix)  # 距离矩阵的行数和列数
shorter_path = np.random.choice(row, row, replace=False)  # 随机生成路径初始解---不重复
shorter_dist = ls.calculate_path_distance(ls.dist_matrix, shorter_path)  # 计算初始解的路径长度
while iterx < iterMax:
    local_path = np.random.choice(row, row, replace=False)  # 预先随机生成路径
    [current_dist, current_path] = ls.VND(ls.dist_matrix, local_path)  # VND 算法求解
    if current_dist < shorter_dist:  # 如果找到更短的路径，则更新
        shorter_dist = current_dist
        shorter_path = current_path
        iterx = 0  # 更新后重置迭代次数
    else:
        iterx += 1
print("最短路径长度：", shorter_dist)
print("最短路径：", shorter_path)
