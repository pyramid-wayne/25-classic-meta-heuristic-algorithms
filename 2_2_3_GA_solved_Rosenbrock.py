"""
遗传算法求解Rosenbrock函数;GA: Genetic Algorithm; Rosenbrock Function: (1-x1)^2 + 100(x2-x1^2)^2
"""
from HeuristicAlgorithms import GA
import numpy as np

GA = GA()
popsize = 200  # 种群大小
MaxIter = 1000  # 最大迭代次数
p_c = 0.8  # 交叉概率
p_m = 0.01  # 变异概率
Length1 = 10  # 变量x1的编码长度
Length2 = 10  # 变量x2的编码长度
ChromLength = Length1 + Length2  # 编码总长度
IterNum = 0  # 记录迭代次数
Population = GA.GenerateInitialPopulation(ChromLength, popsize)  # 初始化种群编码
[Fitness, x1, x2] = GA.CalculateFitnessValue(popsize, Length1, Length2, Population)  # 计算适应度值
[CurrentBest, BestIndex] = GA.get_max_and_index(Fitness)  # 记录当前最优适应度值和最优个体下标
BestIndividual = Population[BestIndex, :]  # 记录当前最优个体
BestValue = CurrentBest  # 记录当前最优适应度值
best_x1, best_x2 = 0.0, 0.0
# 开始迭代求解
while IterNum < MaxIter:
    IterNum += 1
    print("第%d次迭代，当前最优适应度值为%f" % (IterNum, CurrentBest))
    # [Fitness, x1, x2] = GA.CalculateFitnessValue(popsize, Length1, Length2, Population)  # 计算适应度值
    Population = GA.SelectRouletteWheel(Population, Fitness, popsize)  # 轮盘赌选择新一代种群
    Population = GA.CrossoverOperator(popsize, Population, ChromLength, p_c)  # 交叉操作
    Population = GA.MutationOperator(popsize, Population, ChromLength, p_m)  # 变异操作
    [Fitness, x1, x2] = GA.CalculateFitnessValue(popsize, Length1, Length2, Population)  # 计算新一代适应度值
    [CurrentBest, BestIndex] = GA.get_max_and_index(Fitness)  # 记录当前最优适应度值和最优个体下标
    if CurrentBest >= BestValue:
        BestValue = CurrentBest
        best_x1 = x1
        best_x2 = x2
print("最优适应度值为%f", BestValue)
print("最优变量值为x1=%f,x2=%f" % (best_x1, best_x2))
