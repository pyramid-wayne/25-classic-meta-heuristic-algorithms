# 禁忌搜索算法，解决旅行商问题;TS: Tabu Search, TSP: Traveling Salesman Problem

import numpy as np
from HeuristicAlgorithms import TabuSearch as TS

ts = TS()
# -------- 参数设置 --------
city_num = ts.tsp_coord.shape[1]  # 城市数量
tabu_size = np.ceil(city_num ** 0.5).astype(int)  # 禁忌表长度
candidate_num = 2 * city_num  # 每次迭代候选解个数---邻域解数量，不大于n*(n-1)/2
max_iter = 200  # 最大迭代次数
# -------- 初始化 --------
sym_matrix = ts.calculate_coord_dist_symmetry_matrix(ts.tsp_coord)  # 计算距离对称矩阵
tabu_list = np.zeros((city_num, city_num))  # 初始化禁忌表
be_cands_num = 6  # 邻域解集数量
best_fitness_value = 1e10  # 初始化最优适应度值
init_solution = np.random.permutation(city_num)  # 随机生成初始解 np.random.choice(city_num, city_num, replace=False)
be_cands = np.ones((be_cands_num, 4))  # 初始化邻域解集：邻域集标号、邻域解距离、邻域距离、邻域交换的两个城市编号

best_solution = init_solution  # 记录最优解
current_solution = init_solution  # 初始化当前解
cands_list = np.zeros((candidate_num, city_num), dtype=np.int32)  # 记录邻域解集
current_time = 0
candidate_fitness = np.zeros(candidate_num)  # 记录候选解适应度，保存候选解
# -------- 迭代搜索 --------
update_flag = 0  # 更新标志位
while current_time < max_iter:
    neibor_set = ts.generate_neiborhood_set(city_num, candidate_num)  # 返回一组不重复的邻域交换位置
    for i in range(candidate_num):
        cands_list[i, :] = current_solution.copy()
        cands_list[i, neibor_set[i, 0]], cands_list[i, neibor_set[i, 1]] = cands_list[i, neibor_set[i, 1]], cands_list[
            i, neibor_set[i, 0]]  # 交换两个城市
        candidate_fitness[i] = ts.calculate_path_distance(sym_matrix, cands_list[i, :])  # 计算候选解适应度
    # fitness 排序 小到大
    [fitness_value, fitness_index] = ts.sort_arr_indices(candidate_fitness)  # 返回排序后的值、索引
    for i in range(be_cands_num):  # 整理be_cands_num个最优邻域解
        be_cands[i, 0] = int(fitness_index[i])  # 记录邻域解标号
        be_cands[i, 1] = fitness_value[i]  # 记录邻域解距离
        be_cands[i, 2] = neibor_set[fitness_index[i], 0]  # 记录邻域交换的两个城市编号
        be_cands[i, 3] = neibor_set[fitness_index[i], 1]
    if be_cands[0, 1] < best_fitness_value:  # 更新最优解----无条件接受
        best_fitness_value = be_cands[0, 1]  # 更新最优适应度值
        current_solution = cands_list[int(be_cands[0, 0]), :]  # 更新当前解
        best_solution = current_solution.copy()  # 更新最优解
        update_flag = 1  # 更新标志位
        tabu_list = ts.update_tabu_list(tabu_list, int(be_cands[0, 2]), int(be_cands[0, 3]), city_num, tabu_size)  # 更新禁忌表
    else:  # 接受劣解----接受条件：邻域解不在禁忌表中
        for i in range(be_cands_num):
            if tabu_list[int(be_cands[i, 2]), int(be_cands[i, 3])] == 0:  # 邻域解不在禁忌表中
                current_solution = cands_list[int(be_cands[i, 0]), :]  # 更新当前解
                tabu_list = ts.update_tabu_list(tabu_list, int(be_cands[i, 2]), int(be_cands[i, 3]), city_num, tabu_size)  # 更新禁忌表
                update_flag = 1  # 更新标志位
                break
    current_time += 1  # 迭代次数加1
    if update_flag == 1:  # 更新标志位 --- 发生更新
        # 绘制图像
        print('iter:', current_time, 'best_fitness_value:', best_fitness_value)
        update_flag = 0  # 重置标志位
print('最优解：', best_solution)
print('最优适应度值：', best_fitness_value)
