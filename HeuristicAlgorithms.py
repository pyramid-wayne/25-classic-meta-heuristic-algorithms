# 《25个经典元启发算法---从设计到MATLAB实现》 ----- Python 实现
import numpy as np
import matplotlib.pyplot as plt


class HeuristicAlgorithm:
    def __init__(self):
        print('Initializing Heuristic Algorithm')
        self.dist_matrix = self.get_countries_dist_matrix()  # 获取城市距离矩阵
        self.gr17_matrix = self.get_gr17_dist_matrix()  # 获取gr17距离矩阵
        self.tsp_coord = self.get_tsp_coord()  # 获取TSP坐标

    @staticmethod
    def get_countries_dist_matrix():
        dist_matrix = np.array([
            [0, 350, 290, 670, 600, 500, 660, 440, 720, 410, 480, 970],
            [350, 0, 340, 360, 280, 375, 555, 490, 785, 760, 700, 1100],
            [290, 340, 0, 580, 410, 630, 795, 680, 1030, 695, 780, 1300],
            [670, 360, 580, 0, 260, 380, 610, 805, 870, 1100, 1000, 1100],
            [600, 280, 410, 260, 0, 610, 780, 735, 1030, 1000, 960, 1300],
            [500, 375, 630, 380, 610, 0, 160, 645, 500, 950, 815, 950],
            [660, 555, 795, 610, 780, 160, 0, 495, 345, 820, 680, 830],
            [440, 490, 680, 805, 735, 645, 495, 0, 350, 435, 300, 625],
            [720, 785, 1030, 870, 1030, 500, 345, 350, 0, 475, 320, 485],
            [410, 760, 695, 1100, 1000, 950, 820, 435, 475, 0, 265, 745],
            [480, 700, 780, 1000, 960, 815, 680, 300, 320, 265, 0, 585],
            [970, 1100, 1300, 1100, 1300, 950, 830, 625, 485, 745, 585, 0]
        ])
        return dist_matrix

    @staticmethod
    def get_gr17_dist_matrix():
        """
        获取gr17距离矩阵
        :return: gr17距离矩阵
        """
        gr17Matrix = np.array([
            [0, 633, 257, 91, 412, 150, 80, 134, 259, 505, 353, 324, 70, 211, 268, 246, 121],
            [633, 0, 390, 661, 227, 488, 572, 530, 555, 289, 282, 638, 567, 466, 420, 745, 518],
            [257, 390, 0, 228, 169, 112, 196, 154, 372, 262, 110, 437, 191, 74, 53, 472, 142],
            [91, 661, 228, 0, 383, 120, 77, 105, 175, 476, 324, 240, 27, 182, 239, 237, 84],
            [412, 227, 169, 383, 0, 267, 351, 309, 338, 196, 61, 421, 346, 243, 199, 528, 297],
            [150, 488, 112, 120, 267, 0, 63, 34, 264, 360, 208, 329, 83, 105, 123, 364, 35],
            [80, 572, 196, 77, 351, 63, 0, 29, 232, 444, 292, 297, 47, 150, 207, 332, 29],
            [134, 530, 154, 105, 309, 34, 29, 0, 249, 402, 250, 314, 68, 108, 165, 349, 36],
            [259, 555, 372, 175, 338, 264, 232, 249, 0, 495, 352, 95, 189, 326, 383, 202, 236],
            [505, 289, 262, 476, 196, 360, 444, 402, 495, 0, 154, 578, 439, 336, 240, 685, 390],
            [353, 282, 110, 324, 61, 208, 292, 250, 352, 154, 0, 435, 287, 184, 140, 542, 238],
            [324, 638, 437, 240, 421, 329, 297, 314, 95, 578, 435, 0, 254, 391, 448, 157, 301],
            [70, 567, 191, 27, 346, 83, 47, 68, 189, 439, 287, 254, 0, 145, 202, 289, 55],
            [211, 466, 74, 182, 243, 105, 150, 108, 326, 336, 184, 391, 145, 0, 57, 426, 96],
            [268, 420, 53, 239, 199, 123, 207, 165, 383, 240, 140, 448, 202, 57, 0, 483, 153],
            [246, 745, 472, 237, 528, 364, 332, 349, 202, 685, 542, 157, 289, 426, 483, 0, 336],
            [121, 518, 142, 84, 297, 35, 29, 36, 236, 390, 238, 301, 55, 96, 153, 336, 0]
        ])
        return gr17Matrix

    @staticmethod
    def get_tsp_coord():
        """
        获取TSP坐标
        :return: TSP坐标
        """
        return np.array([
            [8.54, 0.77, 17.02, 0.55, 18.47, 0.61, 10.36, 8.39, 4.85, 17.08, 3.38, 9.59, 7.01, 16.62, 10.84, 2.58, 5.02,
             5.78, 17.33, 7.43],
            [4.15, 2.52, 4.41, 12.03, 0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65, 5.22, 9.67,
             16.23, 6.34, 6.51, 0.55]
        ])

    @staticmethod
    def get_max_and_index(arr):
        max_val = np.max(arr)
        flat_idx = np.argmax(arr)
        coord = np.unravel_index(flat_idx, arr.shape)
        return max_val, coord

    @staticmethod
    def sort_list_indices(lst):
        """
        输入：列表 lst
        输出：排序后的列表sorted_lst，排序对应的原始索引sorted_indices
        """
        sorted_pairs = sorted(enumerate(lst), key=lambda x: x[1])
        sorted_indices = [i for i, v in sorted_pairs]
        sorted_lst = [v for i, v in sorted_pairs]
        return sorted_lst, sorted_indices

    @staticmethod
    def sort_arr_indices(data, descending=False):
        """
        排序一个列表或一维数组，返回排序后的值和对应的原始索引。
        
        参数：
            data (list or np.ndarray): 输入数据
            descending (bool): 是否降序排序（默认升序）
        
        返回：
            sorted_data: 排序后的值
            sorted_indices: 对应原始索引
        """
        arr = np.array(data)
        indices = np.argsort(arr)

        if descending:
            indices = indices[::-1]

        sorted_arr = arr[indices]
        return sorted_arr, indices

    def flatten(self, lst):
        """
        展平嵌套列表
        :param lst: 嵌套列表
        :return: 展平后的列表
        """
        result = []
        for item in lst:
            if isinstance(item, (list, tuple, np.ndarray)):
                result.extend(self.flatten(item))  # 递归展开
            else:
                result.append(item)
        return result

    @staticmethod
    def get_min_and_index_nd(arr):
        """
        获取多维数组的最小值及其索引
        :param arr: 多维数组
        :return: 最小值, 索引
        """
        min_val = np.min(arr)
        min_idx = np.unravel_index(np.argmin(arr), arr.shape)
        return min_val, min_idx

    @staticmethod
    def get_min_and_index(arr):
        """
        获取一维数组的最小值及其索引
        :param arr: 一维数组
        :return: 最小值, 索引
        """
        return np.min(arr), np.argmin(arr)

    @staticmethod
    def calculate_path_length(dis_matrix, path):
        """
        计算路径长度----path的大小为city_num+1
        :param dis_matrix: 距离矩阵
        :param path: 路径
        :return: 路径长度
        """
        return sum(dis_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))

    @staticmethod
    def calculate_coord_dist_symmetry_matrix(coord):
        """
        计算坐标距离对称矩阵
        :param coord: 坐标
        :return: 距离对称矩阵
        """
        [row, col] = coord.shape
        dis_matrix = np.zeros((col, col))
        for i in range(col):
            for j in range(i, col):
                dis_matrix[i, j] = np.sqrt((coord[0, i] - coord[0, j]) ** 2 + (coord[1, i] - coord[1, j]) ** 2)
                dis_matrix[j, i] = dis_matrix[i, j]
        return dis_matrix

    @staticmethod
    def calculate_path_distance(dist_matrix, path):
        """
        计算单条路径距离： path为city_num大小，最后做一次收尾连接
        :param dist_matrix: 距离矩阵
        :param path: 路径
        :return: 距离
        """
        distance = 0
        for i in range(len(path) - 1):
            distance += dist_matrix[path[i]][path[i + 1]]
        distance += dist_matrix[path[-1]][path[0]]
        return distance

    def calculate_all_paths_distance(self, dist_matrix, paths):
        """
        计算所有路径距离
        :param dist_matrix: 距离矩阵
        :param paths: 所有路径
        :return: 所有路径距离
        """
        distances = []
        for path in paths:
            distances.append(self.calculate_path_distance(dist_matrix, path))
        return distances
    
    @ staticmethod
    def animate_plt_tsp(plt_path,tsp_coord,best_distance,iter_count):
        """
        动态绘制TSP路径
        :return:
        """
        # 绘制图像---最佳路径
        plt.clf()
        plt.plot(tsp_coord[0, plt_path], tsp_coord[1, plt_path], 'r.-')  # 绘制最优解路径
        plt.title('Iteration: '+str(iter_count)+' TSP Path Distance: ' + str(best_distance))
        plt.xlabel('city_x')
        plt.ylabel('city_y')
        plt.pause(0.1)  # 暂停0.1秒



class GeneticAlgorithm(HeuristicAlgorithm):
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
        best_value = -1000000
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

    @staticmethod
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

    @staticmethod
    def DecodeChromosome(Population, point, length, j):
        """
        解码染色体 二进制转换
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


class EvolutionStrategy(HeuristicAlgorithm):
    """
    进化策略：
    ES个体强调受控的自然变异方式；GA则偏重于随机的变化；
    ES强调的是种群中个体之间的相互竞争；GA强调的是个体之间的相互合作；
    ES的个体由两部分实数编码值构成，第一部分实数基因表示解的变量值，第二部分实数基因表示该个体的权值；GA基因值的改变则按照交叉概率和变异概率进行。
    ES新品种的选择多采用精英策略，即保留上一代中的最优个体，GA则采用轮盘赌选择策略。
    """

    # 初始化函数
    def __init__(self):
        # 调用父类的初始化函数
        super().__init__()

    @staticmethod
    def binaryfunc(x1, x2):
        """
        二进制函数
        :param x1: x1
        :param x2: x2
        :return: Fitness 适应度
        """
        y = 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)
        return y


class EvolutionaryProgramming(HeuristicAlgorithm):
    """
    进化编程：
    ES个体强调受控的自然变异方式；GA则偏重于随机的变化；
    ES强调的是种群中个体之间的相互竞争；GA强调的是个体之间的相互合作；
    ES的个体由两部分实数编码值构成，第一部分实数基因表示解的变量值，第二部分实数基因表示该个体的权值；GA基因值的改变则按照交叉概率和变异概率进行。
    ES新品种的选择多采用精英策略，即保留上一代中的最优个体，GA则采用轮盘赌选择策略。
    EP与ES的不同点：
    1. 种群初始化：EP采用随机初始化，ES采用启发式初始化；
    2. 适应度函数：EP采用目标函数值，ES采用目标函数值与约束条件的乘积；
    3. 交叉操作：EP采用单点交叉，ES采用多点交叉；
    4. 变异操作：EP采用单点变异，ES采用多点变异；
    5. 选择操作：EP采用轮盘赌选择，ES采用锦标赛选择；
    6. 算法终止条件：EP采用最大迭代次数，ES采用最大迭代次数或最优解不再变化。
    --------
    7. ES有可选的重组操作，EP只强调变异操作。
    8. ES的变异以随机方式进行，EP每个个体都要参与变异
    9. ES产生λ个子代λ>=μ，从中选择μ个子代；EP只产生μ个子代。`
    """

    # 初始化函数
    def __init__(self):
        # 调用父类的初始化函数
        super().__init__()

    @staticmethod
    def binary_func(x1, x2):
        """
        二进制函数
        :param x1: x1
        :param x2: x2
        :return: Fitness 适应度
        """
        y = 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)
        return y

    def compute_fitness(self, x1_mat, x2_mat):
        """
        计算适应度
        :param x1_mat: x1矩阵
        :param x2_mat: x2矩阵
        :return: fitness 适应度
        """
        fitness = np.zeros(len(x1_mat))
        for i in range(len(x1_mat)):
            fitness[i] = self.binary_func(x1_mat[i], x2_mat[i])
        return fitness


class DifferentialEvolution(HeuristicAlgorithm):
    """
    差分进化算法：
    DE算法是一种基于群体搜索的启发式优化算法，它通过群体中个体之间的相互协作和竞争来寻找最优解。
    DE算法的基本思想是通过差分操作来产生新的个体，然后通过选择操作来更新种群中的个体。
    DE算法的主要优点是简单、易于实现、鲁棒性强、收敛速度快等。
    DE算法的主要缺点是容易陷入局部最优解、对参数的选择敏感等。
    DE算法是一种改进的差分进化算法，它通过引入自适应参数来提高算法的性能。
    -------------------------------------------------------------
    主要思想是使用向量差来扰动向量种群，通过交叉、变异和选择操作，迭代优化目标解；
    步骤： 初始化种群 -> 计算适应度 -> 产生新种群 -> 选择操作 -> 更新种群 -> 判断终止条件 -> 返回最优解
    初始种群：随机生成一组个体，每个个体代表一个解；
    随机选择两个个体向量：从种群中随机选择两个个体向量，作为差分向量；
    计算差分向量：计算差分向量；
    产生新个体：将差分向量*权重与一个个体向量相加，得到一个新个体；
    交叉操作：将新个体与个体向量进行交叉操作，得到一个新个体；
    变异操作：将新个体与个体向量进行变异操作，得到一个新个体；
    选择操作：将新个体与个体向量进行比较，选择适应度较高的个体；
    """

    # 初始化函数
    def __init__(self):
        # 调用父类的初始化函数
        super().__init__()

    @staticmethod
    def rosenbrock(x):
        """
        Rosenbrock函数
        :param x: [x1, x2]
        :return: Fitness 适应度
        """
        y = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        return y


class EstimationOfDistributionAlgorithm(HeuristicAlgorithm):
    """
    分布估计算法：
    EDA算法是一种基于概率的优化算法，它通过模拟自然界中的生物进化过程来寻找最优解。
    EDA算法的基本思想是通过估计解的分布来产生新的解，然后通过选择操作来更新解的分布。
    EDA算法的主要优点是简单、易于实现、鲁棒性强、收敛速度快等。
    EDA算法的主要缺点是容易陷入局部最优解、对参数的选择敏感等。
    EDA算法是一种改进的分布估计算法，它通过引入自适应参数来提高算法的性能。
    -------------------------------------------------------------
    主要思想，是使用概率模型来模拟解的分布，通过交叉、变异和选择操作，迭代优化目标解；
    步骤： 初始化种群 -> 计算适应度 -> 产生新种群 -> 选择操作 -> 更新种群 -> 判断终止条件 -> 返回最优解
    初始种群：随机生成一组个体，每个个体代表一个解；
    计算适应度：计算每个个体的适应度；
    产生新种群：根据适应度，使用概率模型产生新的个体；
    选择操作：将新个体与个体向量进行比较，选择适应度较高的个体；
    """

    def __init__(self):
        super().__init__()
        self.weight = np.array(
            [382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150, 823460, 903959, 853665, 551830,
             610856, 670702, 488960, 951111, 323046, 446298, 931161, 313859, 496951, 264724, 224916, 169684])
        self.value = np.array(
            [825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457, 1679693, 1902996, 1844992, 1049289,
             1252836, 1319836, 953277, 2067538, 675367, 853655, 1826027, 65731, 901489, 577242, 466257, 369262])

    # @staticmethod


class LocalSearch(HeuristicAlgorithm):
    """
    领域搜索系列算法：从当前解出发，产生一组邻域候选解，并依照某种策略接受其中满足条件的解。
    1.VNS: variable neighborhood search, 变邻域搜索算法
    2.ILS: iterative local search, 迭代局部搜索算法
    3.GLS: guided local search, 引导局部搜索算法
    4.GRASP: greedy randomized adaptive search procedure, 贪婪随机自适应搜索算法
    """

    def __init__(self):
        super().__init__()

    def VND(self, dist_matrix, path):
        """
        变邻域搜索算法---VND：variable neighborhood search
        三种邻域结构展开搜索
        :param dist_matrix: 距离矩阵
        :param path: 初始路径
        :return: 最优距离，最优路径
        """
        dis, k, lmax, i = 1e20, -1, 3, 1  # 初始化最优距离，最优路径，邻域结构数，邻域结构索引
        while i <= lmax:
            # 产生领域
            if i == 1:
                paths = self.vnd_swap(path)  # swap 算子
            elif i == 2:
                paths = self.vnd_two_opt_swap(path)  # two_opt_swap 算子
            elif i == 3:
                paths = self.vnd_two_h_opt_swap(path)  # two_h_opt_swap 算子
            # 计算距离
            for vnd_path in paths:
                vnd_dis = self.calculate_path_distance(dist_matrix, vnd_path)
                if vnd_dis < dis:
                    dis = vnd_dis
                    path = vnd_path
                    k = 1
            if k == 1:  # 如果找到更优解，则重新搜索第一个邻域结构
                i = 1
                k = -1
            else:  # 否则搜索下一邻域结构
                i += 1
        return dis, path

    @staticmethod
    def vnd_swap(path):
        """
        VND 算法中的 swap 算子
        :param path: 初始路径
        :return: 交换后的路径
        """
        length = len(path)
        count = np.arange(1, length, 1)
        neighbor_num = np.sum(count).astype(int)  # 邻域个数
        neighbor_paths = np.zeros((neighbor_num, length), dtype=int)  # 邻域路径
        k = 0
        for i in range(length - 1):
            for j in range(i + 1, length):
                s = path.copy()  # 复制初始路径
                s[i], s[j] = s[j], s[i]  # 交换路径中两个城市
                neighbor_paths[k, :] = s  # 保存邻域路径
                k += 1  # 邻域路径索引
        return neighbor_paths

    @staticmethod
    def vnd_two_opt_swap(path):
        """
        VND 算法中的 two_opt_swap 算子 区间中的城市逆序交换
        :param path: 初始路径
        :return: 交换后的路径
        """
        length = len(path)
        step = 3
        count = np.arange(1, length - step + 1, 1)
        neighbor_num = np.sum(count).astype(int)  # 邻域个数
        neighbor_paths = np.zeros((neighbor_num, length), dtype=int)  # 邻域路径
        k = 0
        for i in range(length):
            for j in range(i + step, length):
                s = path.copy()  # 复制初始路径
                s[i:j + 1] = s[i:j + 1][::-1]  # 交换路径中的城市 i:j+1  [::-1] 表示逆序
                neighbor_paths[k, :] = s  # 保存邻域路径
                k += 1
        return neighbor_paths

    def vnd_two_h_opt_swap(self, path):
        """
        VND 算法中的 two_h_opt_swap 算子： 选中的城市放置在前
        :param path: 初始路径
        :return: 交换后的路径
        """
        length = len(path)
        count = np.arange(1, length, 1)
        neighbor_num = np.sum(count).astype(int)  # 邻域个数
        neighbor_paths = np.zeros((neighbor_num, length), dtype=int)  # 邻域路径
        k = 0
        for i in range(length):
            for j in range(i + 1, length):
                s = path.copy()  # 复制初始路径
                lst = [s[i], s[j], s[0:i], s[i + 1:j], s[j + 1:length]]  # i,j序号城市放置在前
                s = self.flatten(lst)
                neighbor_paths[k, :] = s  # 保存邻域路径
                k += 1
        return neighbor_paths

    @staticmethod
    def ILS():
        # ILS是一种用于解决组合优化问题的启发式算法，它结合了局部搜索和随机扰动策略。
        # 在TSP问题中，ILS算法通过构造一个初始解，然后通过局部搜索和随机扰动来改进解，从而找到问题的近似最优解。
        # ILS算法的主要步骤包括：
        # 1. 构造一个初始解：通过贪心策略构造一个初始解，即从一个城市开始，每次选择离当前城市最近的城市作为下一个城市，直到访问所有城市。
        # 2. 局部搜索：在初始解的基础上，通过局部搜索算法（如VND）找到初始解的邻域中的最优解。
        # 3. 随机扰动：在局部搜索找到的最优解的基础上，通过随机扰动生成一个新解，即随机选择一个城市，然后将其与当前解中的某个城市交换位置，生成一个新的解。
        # 4. 重复步骤2和步骤3，直到达到终止条件，如达到最大迭代次数或解的改进幅度小于某个阈值。

        # ILS算法的参数包括：
        # 1. 最大迭代次数：用于控制算法的终止条件，即算法在达到最大迭代次数后停止。
        # 2. 解的改进幅度阈值：用于控制算法的终止条件，即当解的改进幅度小于阈值时，算法停止。
        pass

    @staticmethod
    def ils_griewank(x_mat):
        """
        ILS 算法中的 Griewank 函数
        :param x_mat: 待优化参数[x0,x1,...,xn]
        :return: 函数值
        """
        dim = x_mat.shape[0]
        sum = 0
        prod = 1
        for i in range(dim):
            sum += x_mat[i] ** 2
            prod *= np.cos(x_mat[i] / np.sqrt(i + 1))
        return 1 + sum / 4000 - prod

    @staticmethod
    def GRASP():
        # GRASP是一种用于解决组合优化问题的启发式算法，它结合了贪心算法和随机搜索策略。
        # 在TSP问题中，GRASP算法通过构造一个初始解，然后通过局部搜索和随机扰动来改进解，从而找到问题的近似最优解。
        # GRASP算法的主要步骤包括：
        # 1. 构造一个初始解：通过贪心策略构造一个初始解，即从一个城市开始，每次选择离当前城市最近的城市作为下一个城市，直到访问所有城市。
        # 2. 生成一个局部解：在初始解的基础上，通过随机扰动生成一个局部解，即随机选择一个城市，然后将其与当前解中的某个城市交换位置，生成一个新的解。
        # 3. 选择最优解：在所有生成的局部解中，选择最优解作为新的当前解。
        # 4. 重复步骤2和步骤3，直到达到终止条件，如达到最大迭代次数或解的改进幅度小于某个阈值。

        # GRASP算法的参数包括：
        # 1. α：用于控制随机扰动的参数，α越大，随机扰动的概率越大。
        # 2. 最大迭代次数：用于控制算法的终止条件，即算法在达到最大迭代次数后停止。
        # 3. 解的改进幅度阈值：用于控制算法的终止条件，即当解的改进幅度小于阈值时，算法停止。
        pass

    def grasp_rcl(self, dist_matrix, alpha, prob_size):
        """
        GRASP 算法中的 RCL 函数
        :param dist_matrix: 距离矩阵
        :param alpha: α 
        :param prob_size:  城市数量
        :return: RCL 集合
        """
        start = np.random.choice(prob_size, 1)[0]  # 随机选择一个城市作为起点
        rcl_path = [start]  # 记录搜索路径
        while len(rcl_path) < prob_size:  # 当搜索路径未遍历完所有城市时，继续搜索
            random_num = np.random.rand(1)
            if random_num > alpha:  # 如果随机数大于α，则选择距离最近的城市
                city = rcl_path[-1]  # 获取当前城市
                city_dist = dist_matrix[city, :]  # 获取当前城市到其他城市的距离
                city_dist[city] = 1e9  # 将当前城市到自身的距离设为无穷大
                next_city = -1
                next_value = 1e9
                for i in range(len(city_dist)):
                    if city_dist[i] < next_value and i not in rcl_path:  # 如果当前城市到其他城市的距离小于下一个城市的距离，且该城市不在搜索路径中
                        next_city = i
                        next_value = city_dist[i]
                rcl_path.append(next_city)
            else:  # 如果随机数小于等于α，则选择距离在α范围内的城市
                next_city = np.random.choice(prob_size, 1)[0]  # 随机选择一个城市
                while next_city in rcl_path:  # 如果随机选择的城市已经在搜索路径中，则重新选择
                    next_city = np.random.choice(prob_size, 1)[0]
                rcl_path.append(next_city)
        # path.append(start)     # 将起点添加到搜索路径的末尾，形成闭环
        rcl_path = np.array(rcl_path)
        rcl_value = self.calculate_path_distance(dist_matrix, rcl_path)
        return rcl_value, rcl_path

    def grasp_local_search(self, dist_matrix, path):
        """
        GRASP 算法中的局部搜索
        :param dist_matrix: 距离矩阵
        :param path: 搜索路径
        :return: 最优解
        """
        path = np.array(path)
        current_best_path = path.copy()
        best_route = path.copy()
        best_value = self.calculate_path_distance(dist_matrix, path)
        matrix_size = len(path) - 1  # -1  取决于 path是否为闭环
        for i in range(matrix_size - 2):  # 循环邻域交换搜索
            for j in range(i + 1, matrix_size - 1):
                best_route[i:j + 1] = best_route[i:j + 1][::-1]  # 交换城市顺序
                cal_value = self.calculate_path_distance(dist_matrix, best_route)
                if cal_value < best_value:
                    best_value = cal_value
                    current_best_path = best_route.copy()
                best_route = path.copy()
        return best_value, current_best_path


class SimulatedAnnealing(HeuristicAlgorithm):
    # SA是一种用于解决组合优化问题的启发式算法，它结合了模拟退火和局部搜索策略。
    # 在TSP问题中，SA算法通过构造一个初始解，然后通过模拟退火和局部搜索来改进解，从而找到问题的近似最优解。
    # SA算法的主要步骤包括：
    # 1. 构造一个初始解：通过贪心策略构造一个初始解，即从一个城市开始，每次选择离当前城市最近的城市作为下一个城市，直到访问所有城市。
    # 2. 模拟退火：在初始解的基础上，通过模拟退火算法找到初始解的邻域中的最优解。
    # 3. 局部搜索：在模拟退火找到的最优解的基础上，通过局部搜索找到初始解的邻域中的最优解。
    # 4. 重复步骤2和步骤3，直到达到终止条件，如达到最大迭代次数或解的改进幅度小于某个阈值。

    # SA算法的参数包括：    
    # 1. 初始温度：用于控制模拟退火的初始温度。
    # 2. 降温速率：用于控制模拟退火的降温速率。
    # 3. 最大迭代次数：用于控制算法的终止条件，即算法在达到最大迭代次数后停止。
    # 4. 解的改进幅度阈值：用于控制算法的终止条件，即当解的改进幅度小于阈值时，算法停止。
    def __init__(self):
        super().__init__()

    @staticmethod
    def generate_neiborhood_set(city_num, neibor_num):
        """
        生成邻居集
        :param city_num: 城市数量
        :param neibor_num: 邻居数量,2-opt 交换
        :return: 邻居集
        """
        ik = 0
        neibor_set = np.zeros((neibor_num, 2), dtype=int)
        while ik < neibor_num:
            part_set = np.random.choice(city_num, 2, replace=False)  # 随机选择两个城市
            part_set = np.sort(part_set)  # 对选择的两个城市进行排序---升序
            flag = 0
            for i in range(neibor_num):
                if neibor_set[i, 0] == part_set[0] and neibor_set[i, 1] == part_set[1]:
                    flag = 1
                    break
            if flag == 0:
                neibor_set[ik, :] = part_set
                ik += 1
            if ik == neibor_num:
                break
        return neibor_set

    def swap_neiborhood(self, current_path, neibor_set):
        """
        交换邻居集
        :param current_path: 当前路径
        :param neibor_set: 邻居集
        :return: 交换后的路径
        """
        city_num = len(current_path)  # 城市数量
        swap_num = neibor_set.shape[0]  # 邻居数量
        swap_path = np.zeros((swap_num, city_num), dtype=int)  # 交换后的路径
        for i in range(swap_num):
            swap_path[i, :] = current_path.copy()
            swap_path[i, neibor_set[i, 0]], swap_path[i, neibor_set[i, 1]] = swap_path[i, neibor_set[i, 1]], swap_path[
                i, neibor_set[i, 0]]  # 交换邻居集中的两个城市
        return swap_path


class TabuSearch(SimulatedAnnealing):
    # TS是一种用于解决组合优化问题的启发式算法，它结合了禁忌搜索和局部搜索策略。
    # 在TSP问题中，TS算法通过构造一个初始解，然后通过禁忌搜索和局部搜索来改进解，从而找到问题的近似最优解。
    # TS算法的主要步骤包括：
    # 1. 构造一个初始解：通过贪心策略构造一个初始解，即从一个城市开始，每次选择离当前城市最近的城市作为下一个城市，直到访问所有城市。
    # 2. 禁忌搜索：在初始解的基础上，通过禁忌搜索算法找到初始解的邻域中的最优解。
    # 3. 局部搜索：在禁忌搜索找到的最优解的基础上，通过局部搜索找到初始解的邻域中的最优解。
    # 4. 重复步骤2和步骤3，直到达到终止条件，如达到最大迭代次数或解的改进幅度小于某个阈值。

    # TS算法的参数包括：
    # 1. 初始解：用于构造初始解。
    # 2. 禁忌表长度：用于控制禁忌搜索的禁忌表长度。
    # 3. 最大迭代次数：用于控制算法的终止条件，即算法在达到最大迭代次数后停止。
    # 4. 解的改进幅度阈值：用于控制算法的终止条件，即当解的改进幅度小于阈值时，算法停止。
    def __init__(self):
        super().__init__()

    @staticmethod
    def update_tabu_list(tabu_list, exchange_x, exchange_y, city_num, tabu_size):
        """
        更新禁忌表
        :param tabu_list: 禁忌表
        :param exchange_x: 交换的城市x
        :param exchange_y: 交换的城市y
        :param city_num: 城市数量
        :param tabu_size: 禁忌表长度
        :return: 更新后的禁忌表
        """
        # 更新禁忌表
        for i in range(city_num):
            for j in range(city_num):
                if tabu_list[i, j] != 0:  # 如果禁忌表中该位置不为0，则减1
                    tabu_list[i, j] = tabu_list[i, j]-1
        tabu_list[exchange_x, exchange_y] = tabu_size
        return tabu_list
    
class ArtificialImmuneSystem(TabuSearch):
    # AIS是一种用于解决组合优化问题的启发式算法，它结合了人工免疫系统和遗传算法策略。
    # 在TSP问题中，AIS算法通过构造一个初始解，然后通过人工免疫系统和遗传算法来改进解，从而找到问题的近似最优解。
    # AIS算法的主要步骤包括：
    # AIS是一种用于解决组合优化问题的启发式算法，它结合了人工免疫系统和遗传算法策略。
    # 在TSP问题中，AIS算法通过构造一个初始解，然后通过人工免疫系统和遗传算法来改进解，从而找到问题的近似最优解。
    # AIS算法的主要步骤包括：
    # 1. 构造一个初始解：通过贪心策略构造一个初始解，即从一个城市开始，每次选择离当前城市最近的城市作为下一个城市，直到访问所有城市。
    # 2. 人工免疫系统：在初始解的基础上，通过人工免疫系统算法找到初始解的邻域中的最优解。
    # 3. 遗传算法：在人工免疫系统找到的最优解的基础上，通过遗传算法找到初始解的邻域中的最优解。
    # 4. 重复步骤2和步骤3，直到达到终止条件，如达到最大迭代次数或解的改进幅度小于某个阈值。

    # AIS算法的参数包括：
    # 1. 初始解：用于构造初始解。
    # 2. 人工免疫系统参数：用于控制人工免疫系统算法的参数，如抗体数量、抗体多样性、抗体亲和度等。
    # 3. 遗传算法参数：用于控制遗传算法的参数，如种群大小、交叉概率、变异概率等。
    # 4. 最大迭代次数：用于控制算法的终止条件，即算法在达到最大迭代次数后停止。
    # 5. 解的改进幅度阈值：用于控制算法的终止条件，即当解的改进幅度小于阈值时，算法停止。
    def __init__(self):
        super().__init__()

class AntColonySystem(ArtificialImmuneSystem):
    # ACS是一种用于解决组合优化问题的启发式算法，它结合了蚁群算法和遗传算法策略。
    # 在TSP问题中，ACS算法通过构造一个初始解，然后通过蚁群算法和遗传算法来改进解，从而找到问题的近似最优解。
    # ACS算法的主要步骤包括：
    # 1. 构造一个初始解：通过贪心策略构造一个初始解，即从一个城市开始，每次选择离当前城市最近的城市作为下一个城市，直到访问所有城市。
    # 2. 蚁群算法：在初始解的基础上，通过蚁群算法找到初始解的邻域中的最优解。
    # 3. 遗传算法：在蚁群算法找到的最优解的基础上，通过遗传算法找到初始解的邻域中的最优解。
    # 4. 重复步骤2和步骤3，直到达到终止条件，如达到最大迭代次数或解的改进幅度小于某个阈值。

    # ACS算法的参数包括：
    # 1. 初始解：用于构造初始解。
    # 2. 蚁群算法参数：用于控制蚁群算法的参数，如蚂蚁数量、信息素浓度、信息素挥发率等。
    # 3. 遗传算法参数：用于控制遗传算法的参数，如种群大小、交叉概率、变异概率等。
    # 4. 最大迭代次数：用于控制算法的终止条件，即算法在达到最大迭代次数后停止。
    # 5. 解的改进幅度阈值：用于控制算法的终止条件，即当解的改进幅度小于阈值时，算法停止。
    def __init__(self):
        super().__init__()

    @ staticmethod
    def MMAS():
        # MMAS是一种用于解决组合优化问题的启发式算法，它结合了蚁群算法和遗传算法策略。
        # 最大最小蚂蚁系统算法MMAS是一种改进的蚁群算法，它通过引入最大最小蚂蚁系统来改进蚁群算法的性能。
        # MMAS算法的主要步骤包括：
        # 1. 构造一个初始解：通过贪心策略构造一个初始解，即从一个城市开始，每次选择离当前城市最近的城市作为下一个城市，直到访问所有城市。
        # 2. 蚁群算法：在初始解的基础上，通过蚁群算法找到初始解的邻域中的最优解。
        # 3. 遗传算法：在蚁群算法找到的最优解的基础上，通过遗传算法找到初始解的邻域中的最优解。
        # 4. 重复步骤2和步骤3，直到达到终止条件，如达到最大迭代次数或解的改进幅度小于某个阈值。

        # MMAS算法的参数包括：
        # 1. 初始解：用于构造初始解。
        # 2. 蚁群算法参数：用于控制蚁群算法的参数，如蚂蚁数量、信息素浓度、信息素挥发率等。
        # 3. 遗传算法参数：用于控制遗传算法的参数，如种群大小、交叉概率、变异概率等。
        # 4. 最大迭代次数：用于控制算法的终止条件，即算法在达到最大迭代次数后停止。
        # 5. 解的改进幅度阈值：用于控制算法的终止条件，即当解的改进幅度小于阈值时，算法停止。
        pass