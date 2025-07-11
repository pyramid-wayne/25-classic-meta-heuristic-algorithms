# 人工免疫系统算法求解旅行商问题，AIS: Artificial Immune System, TSP: Traveling Salesman Problem
import numpy as np
from HeuristicAlgorithms import ArtificialImmuneSystem as AIS
import matplotlib.pyplot as plt


ais = AIS()
# --------- 初始化参数 ---------
city_num = ais.tsp_coord.shape[1]  # 城市数量
D_matrix = ais.calculate_coord_dist_symmetry_matrix(ais.tsp_coord)  # 计算距离对称矩阵
anti_num = 100  # 抗体数量
clone_num = 10  # 克隆数量
max_iter = 100  # 最大迭代次数
cross_rate = 0.1  # 交叉概率
# --------- 初始化抗体群 ---------
solution = np.zeros((city_num, anti_num),dtype=np.int32)
route_len = np.zeros(anti_num)
for i in range(anti_num):
    solution[:, i] = np.random.permutation(city_num)  # 随机生成初始化种群
route_len = ais.calculate_all_paths_distance(D_matrix, solution.T)  # 计算所有抗体路径长度
[route_value, route_index] = ais.sort_arr_indices(route_len)  # 按路径长度排序
best_value = route_value[0]  # 最佳路径长度
best_solution = solution[:, route_index[0]]  # 最佳路径
order_solution = solution[:, route_index]  # 按路径长度排序后的抗体群 --- 排序
trace_length = np.zeros(max_iter)  # 记录每次迭代的最优路径长度
# --------- 迭代优化：人工免疫系统 --------- 
for gen in range(max_iter):
    clone_solution= np.zeros((city_num, np.int32(anti_num / 2)),dtype=np.int32) # 初始化克隆抗体群
    clone_value = np.zeros(np.int32(anti_num / 2)) # 初始化克隆抗体群路径长度
    for i in range(np.int32(anti_num / 2)):  # 前50%抗体，每个克隆clone_num个抗体后完成邻域交换
        #  ------- 克隆操作选择和变异 -------
        a = order_solution[:, i]  # 选择抗体---先转换为列向量复制
        # clone_a = np.repeat(a[:, np.newaxis], clone_num, axis=1)  # 克隆抗体
        # clone_a = np.tile(a[:, np.newaxis], (1, clone_num))  # 克隆抗体
        clone_a = np.repeat(a[:, np.newaxis], clone_num, axis=1)
        for j in range(1,clone_num): # 遍历抗体群,保留初始抗体
            # 随机选择两个位置进行交换
            pos1 = np.random.randint(0, city_num)
            pos2 = np.random.randint(0, city_num)
            while pos1 == pos2:
                pos2 = np.random.randint(0, city_num)
            # 交换位置
            clone_a[pos1, j], clone_a[pos2, j] = clone_a[pos2, j], clone_a[pos1, j]
        #  ------- 克隆抑制 -------
        clone_route_len=ais.calculate_all_paths_distance(D_matrix, clone_a.T)  # 计算新克隆抗体路径长度
        [clone_route_value, clone_route_index] = ais.sort_arr_indices(clone_route_len)  # 按路径长度排序
        clone_solution[:, i] = clone_a[:, clone_route_index[0]]  # 保留最优抗体
        clone_value[i] = clone_route_value[0]  # 保留最优抗体路径长度
    # -------- 刷新种群 --------
    sub_solution= np.zeros((city_num, np.int32(anti_num / 2)),dtype=np.int32)
    sub_value = np.zeros(np.int32(anti_num / 2))
    for i in range(np.int32(anti_num / 2)): # 淘汰部分种群并随机产生替代种群
        sub_solution[:, i] = np.random.permutation(city_num)  # 随机生成初始化种群
        sub_value[i] = ais.calculate_path_distance(D_matrix, sub_solution[:, i])  # 计算所有抗体路径长度
    #  ------- 交叉操作 -------
    combine_solution = np.concatenate((clone_solution, sub_solution), axis=1)  # 合并抗体群
    combine_value = np.concatenate((clone_value, sub_value), axis=0)  # 合并抗体群路径长度
    [combine_route_value, combine_route_index] = ais.sort_arr_indices(combine_value)  # 按路径长度排序
    order_solution = combine_solution[:, combine_route_index]  # 按路径长度排序后的抗体群
    trace_length[gen] = combine_route_value[0]  # 记录每次迭代的最优路径长度
    best_path=order_solution[:,0]
    plt_path = np.append(best_path, best_path[0])
    plt.clf()
    plt.plot(ais.tsp_coord[0, plt_path], ais.tsp_coord[1, plt_path], 'r.-')  # 绘制最优解路径
    plt.title('TSP Path Distance: ' + str(combine_route_value[0]))
    plt.xlabel('city_x')
    plt.ylabel('city_y')
    plt.pause(0.1)  # 暂停0.1秒

print('最优路径长度：', trace_length[-1])
print('最优路径：', order_solution[:, 0])
plt.plot(trace_length, 'r-')
plt.xlabel('iteration')
plt.ylabel('best route length')
plt.title('AIS solution for TSP')
plt.show()

