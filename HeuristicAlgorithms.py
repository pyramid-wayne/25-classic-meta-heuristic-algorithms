# 《25个经典元启发算法---从设计到MATLAB实现》 ----- Python 实现
import numpy as np


class HeuristicAlgorithm:
    def __init__(self):
        print('Initializing Heuristic Algorithm')

    @staticmethod
    def get_max_and_index(arr):
        max_val = np.max(arr)
        flat_idx = np.argmax(arr)
        coord = np.unravel_index(flat_idx, arr.shape)
        return max_val, coord


class GA(HeuristicAlgorithm):
    def __init__(self):
        super().__init__()
        print('Initializing Genetic Algorithm')

    @staticmethod
    def GenerateInitialPopulation(Chromlength, Popsize):
        """
        随机产生初始种群
        :param Chromlength: 染色体长度
        :param Popsize: 种群大小
        :return: pop 种群编码 0 1
        """
        pop = np.zeros([Popsize, Chromlength])  # 初始化种群
        for i in range(Popsize):
            for j in range(Chromlength):
                if np.random.rand() < 0.5:
                    pop[i, j] = 0
                else:
                    pop[i, j] = 1
        return pop

    def CalculateFitnessValue(self, Popsize, Length1, Length2, pop):
        """
        计算适应度---目标函数值
        :param Popsize: 种群大小
        :param Length1: 适应度1的长度
        :param Length2: 适应度2的长度
        :param pop: 种群编码
        :return: fitness 适应度
        """
        best_x1, best_x2 = 0.0, 0.0
        Fitness = np.zeros(Popsize, dtype=float)  # 初始化适应度
        best_value = -1000000000000
        for i in range(Popsize):
            temp1 = self.DecodeChromosome(pop, 0, Length1, i)
            temp2 = self.DecodeChromosome(pop, Length1, Length2, i)
            x1 = 4.096 * temp1 / 1023 - 2.048
            x2 = 4.096 * temp2 / 1023 - 2.048
            Fitness[i] = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
            if Fitness[i] > best_value:
                best_value = Fitness[i]
                best_x1 = x1
                best_x2 = x2
        return Fitness, best_x1, best_x2

    @staticmethod
    def SelectRouletteWheel(pop, Fitness, Popsize):
        """
        轮盘赌：选择下一代种群
        :param pop: 种群编码
        :param Fitness: 适应度
        :param Popsize: 种群大小
        :return: newpop 新种群
        """
        totalFitness = np.sum(Fitness)  # 计算适应度总和
        pFitValues = Fitness / totalFitness  # 计算每个个体的适应度概率
        mFitValues = np.cumsum(pFitValues)  # 计算累积适应度概率
        fitin = 0
        newpop = np.zeros(pop.shape)  # 初始化新种群
        while fitin < Popsize:
            r = np.random.rand()
            for i in range(Popsize):
                if r > mFitValues[i]:  # 找到随机数所在区间
                    continue
                else:
                    newpop[fitin, :] = pop[i, :]  # 将当前个体加入新种群
                    fitin = fitin + 1  # 新种群个体数加1
                    break
        return newpop

    @staticmethod
    def CrossoverOperator(Popsize, pop, Chromlength, P_C):
        """
        交叉操作
        :param Popsize: 种群大小
        :param pop: 种群编码
        :param Chromlength: 染色体长度
        :param P_C: 交叉概率
        :return: newpop 新种群
        """
        half_pop = int(Popsize / 2)
        newpop1 = np.zeros([half_pop, Chromlength])  # 初始化新种群
        newpop2 = np.zeros([half_pop, Chromlength])  # 初始化新种群
        for i in range(half_pop):
            # 随机选择2个交叉点 np.random.permutation(np.arange(1, 11))[:2]
            point = np.random.choice(np.arange(1, Chromlength), size=2, replace=False)
            while point[0] == point[1]:  # 交叉点不能相同
                point = np.random.choice(np.arange(1, Chromlength), size=2, replace=False)
            if point[0] > point[1]:  # 保证point[0]<point[1]
                temp = point[0]
                point[0] = point[1]
                point[1] = temp
            # 交叉操作---取出两个原始体
            temp1 = pop[i, :]
            temp2 = pop[half_pop + i, :]
            # 交叉操作---交叉点之间交换
            p = np.random.rand()
            if p < P_C:
                part1 = temp1[point[0]:point[1] + 1]  # 取出第一个染色体交叉点之间的部分
                part2 = temp2[point[0]:point[1] + 1]  # 取出第二个染色体交叉点之间的部分
                newpop1[i, :] = np.concatenate(
                    [temp1[0:point[0] - 1], part2, temp1[point[1]:]])  # 交叉点之间交换 concatenate column_stack
                newpop2[i, :] = np.concatenate([temp2[0:point[0] - 1], part1, temp2[point[1]:]])  # 交叉点之间交换
            else:
                newpop1[i, :] = temp1
                newpop2[i, :] = temp2
        newpop = np.concatenate((newpop1, newpop2), axis=0)  # 合并两个新种群
        return newpop

    @ staticmethod
    def MutationOperator(Popsize, Population, Chromlength, P_M):
        """
        变异操作
        :param Popsize: 种群大小
        :param Population: 种群编码
        :param Chromlength: 染色体长度
        :param P_M: 变异概率
        :return: newpop 新种群
        """
        for i in range(Popsize):
            p = np.random.rand()  # 产生一个0~1之间的数
            point = np.random.choice(np.arange(0, Chromlength), size=1, replace=False)  # 随机选择一个变异点 0~Chromlength-1
            if p < P_M:
                col_value = int(Population[i, point[0]])
                Population[i, point] = np.bitwise_xor(col_value, 1)  # 与1发生异或操作
        return Population

    @ staticmethod
    def DecodeChromosome(Population, point, length, j):
        """
        解码染色体
        :param Population:      种群编码
        :param point:           起始点
        :param length:          长度
        :param j:               第j个个体
        :return: temp 个体解码后的值
        """
        deci = 0
        for i in range(length):
            deci = deci + Population[j, point + i] * 2 ** (length - i - 1)
        return deci
