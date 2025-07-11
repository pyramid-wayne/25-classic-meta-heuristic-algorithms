# 最大最小蚂蚁系统算法求解旅行商问题，MMAS: Max-Min Ant System, TSP: Traveling Salesman Problem
import numpy as np
from HeuristicAlgorithms import AntColonySystem as ACS
import matplotlib.pyplot as plt

acs = ACS()
# ==========    参数设置     ==========
city_num = acs.tsp_coord.shape[1]  # 城市数量
ant_num = city_num  # 蚂蚁数量
max_iter = 100  # 最大迭代次数
alpha = 1.0  # 信息素重要程度因子
beta = 5.0  # 启发式因子
rho = 0.5  # 信息素挥发因子
route_best = np.zeros((max_iter, city_num), dtype=np.int32)  # 每一代最优路径---各代最佳路径初始化
length_best = 1e20 * np.ones(max_iter)  # 每一代最优路径长度---各代最佳路径长度初始化
tau = np.ones((city_num, city_num))  # 信息素矩阵初始化,tau:残留信息素
tabu = np.zeros((ant_num, city_num), dtype=np.int32)  # 禁忌表---存储路径节点编码，第i只蚂蚁，第j个节点
iter_count = 0  # 迭代次数初始化
city_dists = acs.calculate_coord_dist_symmetry_matrix(acs.tsp_coord)  # 计算城市间距离矩阵+1e-200
sigma = 0.05  # 信息素平滑机制参数
eps = 1.0e-16
eta = np.where(city_dists != 0, 1 / city_dists, 1e20)  # 计算启发式信息素，eta:启发式信息素,np.inf
# ==========    迭代寻优：ant_num放在city_num个节点上     ==========
while iter_count < max_iter:
    # ==========    蚂蚁寻路     ==========
    rand_nodes = np.random.permutation(city_num)  # 随机生成一个0~city_num-1的排列
    tabu[:, 0] = rand_nodes[:ant_num].T  # 将随机生成的排列的前ant_num个节点赋值给禁忌表
    for city_j in range(1, city_num):
        for ant_i in range(ant_num):  # 遍历蚂蚁，构造访问节点的路径
            # ==========    选择下一个访问节点     ==========
            visited = tabu[ant_i, 0:city_j]  # 获取当前蚂蚁已经访问过的城市
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
    # ================ 记录最佳路径 =================
    length_best[iter_count] = np.min(ant_lengths)  # 每一代最优路径长度蚂蚁路径长度最小值
    best_ant_idx = np.argmin(ant_lengths)  # 蚂蚁路径长度最小值的索引
    route_best[iter_count, :] = tabu[best_ant_idx, :]  # 记录每一轮迭代的最佳路径
    global_best_length = np.min(length_best)  # 全局最优路径长度
    global_best_route = route_best[np.argmin(length_best), :]  # 全局最优路径
    # ========== 求出tau_max tau_min信息素界限 ==========
    gb_len = np.min(length_best)  # 全局最优路径长度
    tau_max = 1 / (rho * gb_len)  # 信息素最大值
    p_best = 0.05  # 信息素局部更新概率
    p_best = p_best ** (1 / city_num)  # 信息素局部更新概率
    tau_min = tau_max * (1 - p_best) / ((city_num / 2 - 1) * p_best)  # 信息素最小值
    # ==========  更新信息素矩阵: 采用MMAS信息素更新规则  ==========
    delta_tau = np.zeros((city_num, city_num))  # 信息素增量矩阵
    r0 = 0.5  # 信息素增量矩阵
    if r0 > np.random.rand():
        for city_j in range(city_num - 1):
            # 全局
            delta_tau[global_best_route[city_j], global_best_route[city_j + 1]] += 1 / global_best_length
        delta_tau[global_best_route[-1], global_best_route[0]] += 1 / global_best_length  # 回到出发点
    else:
        for city_j in range(city_num - 1):
            # 局部
            delta_tau[route_best[iter_count, city_j], route_best[iter_count, city_j + 1]] += 1 / length_best[iter_count]
        delta_tau[route_best[iter_count, -1], route_best[iter_count, 0]] += 1 / length_best[iter_count]  # 回到出发点
    tau = (1 - rho) * tau + rho * delta_tau  # 信息素更新
    # 信息素平滑机制
    if iter_count > 3 and length_best[iter_count] == length_best[iter_count - 1] == length_best[iter_count - 2] == \
            length_best[iter_count - 3]:
        for city_j in range(city_num):
            for city_k in range(city_num):
                tau[city_j, city_k] = tau[city_j, city_k] + sigma * (tau_max - tau[city_j, city_k])
    # 信息素截断----限制区间策略，检查信息素是否置于最大最小值之间
    for city_j in range(city_num):
        for city_k in range(city_num):
            if tau[city_j, city_k] > tau_max:
                tau[city_j, city_k] = tau_max
            elif tau[city_j, city_k] < tau_min:
                tau[city_j, city_k] = tau_min
    # ==========  更新禁忌表:清零  ==========
    tabu = np.zeros((ant_num, city_num), dtype=np.int32)  # 初始化禁忌表
    iter_count += 1
    # ================= 绘制结果 =================
    plt_path = np.append(route_best[iter_count - 1, :], route_best[iter_count - 1, 0])  # 将最后一个城市和第一个城市连接起来
    best_distance = np.min(length_best)
    acs.animate_plt_tsp(plt_path, acs.tsp_coord, best_distance, iter_count)
print('最优路径长度：', global_best_length)
print('最优路径：', global_best_route)
