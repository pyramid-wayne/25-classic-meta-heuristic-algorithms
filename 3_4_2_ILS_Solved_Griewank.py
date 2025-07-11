# 迭代局部搜索算法求解Griewank函数,
# ILS:Iterated Local Search,
# Griewank函数: f(x)=1/(4000)^n-1/2*sum(x_i^2)-prod(cos(x_i/sqrt(i))),n=10,[-600,600]^10
import numpy as np
from HeuristicAlgorithms import LocalSearch as LS

ls = LS()
max_iter = 10000  # 最大迭代次数
ndim = 30  # 维度
bound = [-100, 100]  # 搜索空间
ite = 0  # 迭代次数
pop_size = 30  # 种群大小
rdp = 0.5  # 对局部解进行扰动，扰动概率
num_localsearch = 5  # 局部搜索次数
num_perturbation = 10  # 局部搜索后扰动次数
lower_bound = np.zeros((ndim, pop_size))  # 种群下界
upper_bound = np.zeros((ndim, pop_size))  # 种群上界
for i in range(pop_size):
    lower_bound[:, i] = bound[0]
    upper_bound[:, i] = bound[1]
population = lower_bound + np.random.rand(ndim, pop_size) * (upper_bound - lower_bound)  # 初始化种群
f_value = np.zeros(pop_size)  # 种群适应度值
for i in range(pop_size):
    f_value[i] = ls.ils_griewank(population[:, i])  # 计算种群适应度值---个体
[f_value_best, index_best] = ls.get_min_and_index(f_value)  # 种群最优适应度值和最优个体索引
population_best = population[:, index_best]  # 种群最优个体
pre_f_value = f_value  # 上一次迭代种群最优适应度值
# 迭代搜索
while ite < max_iter:
    for i in range(num_localsearch):  # 局部搜索---多次
        a = population_best - 1 / 10 * (population_best - lower_bound[:, i])  # 局部搜索的下限
        b = population_best + 1 / 10 * (upper_bound[:, i] - population_best)  # 局部搜索的上限
        population_new = np.zeros((ndim, num_perturbation))  # 局部随机搜索的种群---初始化
        f_value_new = np.zeros(num_perturbation)  # 局部随机搜索的种群适应度值---初始化
        for j in range(num_perturbation):
            population_new[:, j] = population_best  # 局部随机搜索的种群
            change = (np.random.rand(ndim) < rdp).astype(int)  # 判断是否进行扰动---随机掩膜
            # disturbed = np.where(disturb_code[:, np.newaxis] == 1, data + uniform_noise, data) 
            # result = np.where(mask == 1, data + noise, data)
            noise = a + (b - a) * np.random.rand(1)  # rand(ndim)
            population_new[:, j] = np.where(change == 1, noise, population_new[:, j])  # 局部随机搜索的种群---扰动
            f_value_new[j] = ls.ils_griewank(population_new[:, j])  # 计算局部随机搜索的种群适应度值
        [f_value_new_best, index_new_best] = ls.get_min_and_index(f_value_new)  # 寻找扰动后的最优个体
        if f_value_new_best < f_value_best:  # 判断是否更新最优个体
            f_value_best = f_value_new_best
            population_best = population_new[:, index_new_best]
    print('迭代次数：', ite, '最优适应度值为：', f_value_best)  # , '最优解为：', population_best
    ite += 1
print('最优适应度值为：', f_value_best)
print('最优解为：', population_best)
