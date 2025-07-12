# 模拟退火算法求解TSP问题，SA：Simulated Annealing，TSP：Traveling Salesman Problem
import numpy as np
from HeuristicAlgorithms import SimulatedAnnealing as SA
import matplotlib.pyplot as plt

sa = SA()
city_num = sa.tsp_coord.shape[1]  # 城市数量
T0 = 1000.0  # 初始温度
tau = 0.95  # 降温系数
Ts = 1  # 终止温度
max_inner_loop = 50  # 内循环最大迭代次数
neibor_num = city_num  # 邻域解大小
sym_matrix = sa.calculate_coord_dist_symmetry_matrix(sa.tsp_coord)  # 计算距离对称矩阵
current_path = np.random.choice(city_num, city_num, replace=False)  # 随机生成初始解
current_length = sa.calculate_path_distance(sym_matrix, current_path)  # 计算初始解长度

best_path = current_path.copy()  # 复制初始解为最优解
best_length = current_length.copy()  # 复制初始解长度为最优解长度
current_best_path = current_path.copy()  # 复制初始解为当前最优解
current_best_length = current_length.copy()  # 复制初始解长度为当前最优解长度
# 设置动态绘制最优解路径
plt.ion()
plt.figure()

while T0 > Ts:  # 到达终止温度时停止
    for i in range(max_inner_loop):  # 内循环
        e0 = sa.calculate_path_distance(sym_matrix, current_path)  # 计算当前解长度
        neibor_set = sa.generate_neiborhood_set(city_num, neibor_num)  # 生成邻域解，城市交换组，大小：neibor_num*2
        swap_paths = sa.swap_neiborhood(current_path, neibor_set)  # 交换城市，生成新解
        e2 = sa.calculate_all_paths_distance(sym_matrix, swap_paths)  # 计算新解组长度
        [min_values, min_indexes] = sa.sort_arr_indices(e2)  # 按长度排序--小到大
        e1 = min_values[0]  # 最小长度
        new_path = swap_paths[min_indexes[0]]  # 最小长度对应的新解
        if e1 < e0:  # 如果新解长度小于当前解长度
            current_best_path = new_path.copy()  # 更新当前解
            current_best_length = e1.copy()  # 更新当前解长度
            current_path = new_path.copy()  # 更新当前解
            if e1 < best_length:  # 如果新解长度小于最优解长度
                best_path = current_best_path.copy()  # 更新最优解
                best_length = current_best_length.copy()  # 更新最优解长度
        else:  # 如果新解长度大于当前解长度,按照Metropolis准则判断是否接受新解
            pt = min(1, np.exp((e0 - e1) / T0))  # 计算接受概率
            if np.random.rand() < pt:  # 如果随机数小于接受概率
                current_path = new_path.copy()  # 更新当前解,接受劣解
                e0 = e1.copy()  # 更新当前解长度
    T0 = T0 * tau  # 降温
    plt_path = np.append(best_path, best_path[0])
    plt.clf()
    plt.plot(sa.tsp_coord[0, plt_path], sa.tsp_coord[1, plt_path], 'r.-')  # 绘制最优解路径
    plt.title('TSP Path Distance: ' + str(best_length))
    plt.xlabel('city_x')
    plt.ylabel('city_y')
    plt.pause(0.1)  # 暂停0.1秒

print('最优解长度：', best_length)
print('最优解路径：', best_path)
