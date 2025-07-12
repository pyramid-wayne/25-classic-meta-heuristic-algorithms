# 蚁群系统算法求解TSP问题，ACS：Ant Colony System，TSP：Traveling Salesman Problem
import numpy as np
from HeuristicAlgorithms import AntColonySystem as ACS
import matplotlib.pyplot as plt

acs = ACS()
city_num = acs.tsp_coord.shape[1]  # 城市数量 n
ant_num = city_num  # 蚂蚁数量 m
max_iter = 200  # 最大迭代次数
alpha = 1.0  # 信息素重要程度因子
beta = 5.0  # 启发式因子
rho = 0.5  # 信息素挥发因子
route_best = np.zeros((max_iter, city_num), dtype=np.int32)  # 每一代最优路径---各代最佳路径初始化
length_best = 1e20 * np.ones(max_iter)  # 每一代最优路径长度---各代最佳路径长度初始化
tau = np.ones((city_num, city_num))  # 信息素矩阵初始化,tau:残留信息素
tabu = np.zeros((ant_num, city_num), dtype=np.int32)  # 禁忌表---存储路径节点编码，第i只蚂蚁，第j个节点
iter_count = 0  # 迭代次数初始化
city_dists = acs.calculate_coord_dist_symmetry_matrix(acs.tsp_coord)  # 计算城市间距离矩阵+1e-200
eta = np.where(city_dists != 0, 1 / city_dists, 1e20)  # 计算启发式信息素，eta:启发式信息素,np.inf
print('---------- 参数设置完成 ---------')
while iter_count < max_iter:
    rand_route = np.random.permutation(city_num)  # 随机生成一个路径
    tabu[:, 0] = rand_route[0:ant_num].T  # 将city_num随机数列的前ant_num个作为第一只蚂蚁的路径
    # ================ ant_num只蚂蚁开始搜索 =================
    for city_j in range(1, city_num):  # 每只蚂蚁从第一个城市出发，搜索剩余的j-1个城市
        for ant_i in range(ant_num):  # 每只蚂蚁
            visited = tabu[ant_i, 0:city_j]  # 已访问的城市
            unvisited_p = np.zeros(city_num - city_j)  # 记录未访问节点的选择概率
            unvisited = np.arange(city_num)[~np.isin(np.arange(city_num), visited)]  # 未访问的城市
            # unvisited = unvisited[unvisited != visited]
            threshold_p = 0.5  # 计算未访问城市的概率---阈值
            if np.random.rand() <= threshold_p:  # 如果随机数小于阈值，则选择概率最大的城市
                for city_k in range(len(unvisited)):
                    unvisited_p[city_k] = (tau[visited[-1], unvisited[city_k]] ** alpha) * (
                            eta[visited[-1], unvisited[city_k]] ** beta)
                pos_idx = np.argmax(unvisited_p)  # 最大值索引
                tabu[ant_i, city_j] = unvisited[pos_idx]  # 将选择的城市加入禁忌表
            else:  # 如果随机数大于阈值，则选择随机城市
                for city_k in range(len(unvisited)):
                    unvisited_p[city_k] = (tau[visited[-1], unvisited[city_k]] ** alpha) * (
                            eta[visited[-1], unvisited[city_k]] ** beta)
                unvisited_p = unvisited_p / np.sum(unvisited_p)  # 计算未访问城市的概率---归一化
                cum_p = np.cumsum(unvisited_p)  # 计算未访问城市的概率---累积和
                select_idx = np.where(cum_p >= np.random.rand())[0]
                tabu[ant_i, city_j] = unvisited[select_idx[0]]  # 将选择的城市加入禁忌表
    if iter_count >= 1:  # 第一轮迭代是无需记录
        tabu[0, :] = route_best[iter_count - 1, :]  # 记录每一轮迭代的最佳路径
    # ================ 计算各蚂蚁的路径距离 =================
    ant_lengths = np.zeros(ant_num)  # 蚂蚁路径长度
    for ant_i in range(ant_num):
        ant_route = tabu[ant_i, :]
        for city_j in range(city_num - 1):
            ant_lengths[ant_i] += city_dists[ant_route[city_j], ant_route[city_j + 1]]
        ant_lengths[ant_i] += city_dists[ant_route[city_num - 1], ant_route[0]]  # 蚂蚁的最后一个城市和第一个城市之间的距离
    length_best[iter_count] = np.min(ant_lengths)  # 每一代最优路径长度蚂蚁路径长度最小值
    best_ant_idx = np.argmin(ant_lengths)  # 蚂蚁路径长度最小值的索引
    route_best[iter_count, :] = tabu[best_ant_idx, :]  # 记录每一轮迭代的最佳路径
    # ============ 更新信息素矩阵,采用全局信息素更新规则 =================
    delta_tau = np.zeros((city_num, city_num))  # 初始化信息素增量矩阵
    for city_j in range(city_num - 1):
        # 只在全局最优路径上更新信息素残留
        delta_tau[route_best[iter_count, city_j], route_best[iter_count, city_j + 1]] += 1 / length_best[iter_count]
    # 回到出发点
    delta_tau[route_best[iter_count, city_num - 1], route_best[iter_count, 0]] += 1 / length_best[iter_count]
    tau = (1 - rho) * tau + rho * delta_tau  # 更新信息素矩阵
    # ============ 禁忌表清零 =========
    tabu = np.zeros((ant_num, city_num), dtype=np.int32)
    iter_count += 1  # 迭代次数加1
    # ================= 绘制结果 =================
    plt_path = np.append(route_best[iter_count - 1, :], route_best[iter_count - 1, 0])  # 将最后一个城市和第一个城市连接起来
    best_distance = np.min(length_best)
    acs.animate_plt_tsp(plt_path, acs.tsp_coord, best_distance, iter_count)
print('---------- 迭代完成 ---------')
print('最优路径长度为：', np.min(length_best))
