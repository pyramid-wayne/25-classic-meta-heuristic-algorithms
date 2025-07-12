import random
import math
import operator
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'  # 黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正负号正常显示
# ========== 操作符定义 ==========
# 这里区分一元和二元操作符
binary_ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
}

unary_ops = {
    'sin': math.sin,
    'cos': math.cos,
}


# ========== 表达式树 ==========
class Node:
    def __init__(self, value, children=None):
        self.value = value  # 节点的值
        self.children = children or []  # children 是列表，长度1或2

    def evaluate(self, x):
        """
        计算节点的值
        :param x: 自变量
        :return: 节点的值
        """
        # 如果节点的值在二元操作符中，则返回二元操作符的值
        if self.value in binary_ops:
            return binary_ops[self.value](self.children[0].evaluate(x),
                                          self.children[1].evaluate(x))
        # 如果节点的值在单目操作符中，则返回单目操作符的值
        elif self.value in unary_ops:
            return unary_ops[self.value](self.children[0].evaluate(x))
        # 如果节点的值是x，则返回自变量的值
        elif self.value == 'x':
            return x
        # 否则，返回节点的值
        else:
            return float(self.value)

    def __str__(self):
        """
        将节点转换为字符串
        :return: 节点的字符串表示
        """
        # 如果节点的值是二元操作符，则返回左子节点、操作符和右子节点的字符串表示
        if self.value in binary_ops:
            return f"({str(self.children[0])} {self.value} {str(self.children[1])})"
        # 如果节点的值是一元操作符，则返回操作符和左子节点的字符串表示
        elif self.value in unary_ops:
            return f"{self.value}({str(self.children[0])})"
        # 否则，返回节点的值
        else:
            return str(self.value)

    def copy(self):
        """
        复制节点
        :return: 复制的节点
        """
        # 返回一个新的节点，其值为当前节点的值，子节点为当前节点子节点的复制
        return Node(self.value, [child.copy() for child in self.children])

    def simplify(self):
        """
        简化节点
        :return: 简化后的节点
        """
        # 终端
        if self.value not in binary_ops and self.value not in unary_ops:
            return Node(self.value)

        # 简化子树
        simplified_children = [child.simplify() for child in self.children]

        # 如果所有子节点均为常数，可以提前计算
        if all(child.value not in binary_ops and child.value not in unary_ops for child in simplified_children):
            try:
                vals = [float(child.value) for child in simplified_children]
                if self.value in binary_ops:
                    res = binary_ops[self.value](vals[0], vals[1])
                else:
                    res = unary_ops[self.value](vals[0])
                return Node(str(res))
            except:
                pass

        # 基本规则化简（仅示例，更多规则可以继续加）
        if self.value == '+':
            # x + 0 或 0 + x
            if simplified_children[0].value == '0':
                return simplified_children[1]
            if simplified_children[1].value == '0':
                return simplified_children[0]
        if self.value == '*':
            # 0 * x 或 x * 0
            if simplified_children[0].value == '0' or simplified_children[1].value == '0':
                return Node('0')
            # 1 * x 或 x * 1
            if simplified_children[0].value == '1':
                return simplified_children[1]
            if simplified_children[1].value == '1':
                return simplified_children[0]

        return Node(self.value, simplified_children)


# ========== 树生成 ==========
def generate_random_tree(depth):
    # 递归生成随机树
    if depth == 0 or (depth < 3 and random.random() < 0.3):
        # 随机选择终端
        return Node(random.choice(['x'] + [str(i) for i in range(6)]))
    else:
        # 随机选择一元或二元运算符
        if random.random() < 0.3:
            # 一元操作符
            op = random.choice(list(unary_ops.keys()))
            return Node(op, [generate_random_tree(depth - 1)])
        else:
            # 二元操作符
            op = random.choice(list(binary_ops.keys()))
            return Node(op, [generate_random_tree(depth - 1), generate_random_tree(depth - 1)])


# ========== 适应度 ==========
# 定义一个函数，用于计算适应度
def fitness(ind, x_vals, y_target):
    # 尝试执行以下代码
    try:
        # 计算预测值
        y_pred = [ind.evaluate(x) for x in x_vals]
        # 计算均方误差
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_target, y_pred)) / len(x_vals)
        # 返回均方误差
        return mse
    # 如果发生异常，返回无穷大
    except Exception:
        return float('inf')


# ========== 变异 ==========
# 定义一个函数mutate，用于对树进行变异
def mutate(tree, max_depth=4):
    # 如果随机数小于0.1，则生成一个随机深度的树
    if random.random() < 0.1:
        return generate_random_tree(random.randint(1, max_depth))
    # 如果树的值在binary_ops中，则对树的左右子树进行变异
    elif tree.value in binary_ops:
        return Node(tree.value, [mutate(tree.children[0]), mutate(tree.children[1])])
    # 如果树的值在unary_ops中，则对树的子树进行变异
    elif tree.value in unary_ops:
        return Node(tree.value, [mutate(tree.children[0])])
    # 否则，返回树的副本
    else:
        return tree.copy()


# ========== 交叉 ==========
def crossover(t1, t2):
    # 如果随机数小于0.1，则返回t2的副本
    if random.random() < 0.1:
        return t2.copy()
    # 如果t1和t2的值都在binary_ops中，则返回一个新的节点，其值为t1的值，子节点为t1和t2的子节点进行交叉后的结果
    if t1.value in binary_ops and t2.value in binary_ops:
        return Node(t1.value, [crossover(t1.children[0], t2.children[0]), crossover(t1.children[1], t2.children[1])])
    # 如果t1和t2的值都在unary_ops中，则返回一个新的节点，其值为t1的值，子节点为t1和t2的子节点进行交叉后的结果
    if t1.value in unary_ops and t2.value in unary_ops:
        return Node(t1.value, [crossover(t1.children[0], t2.children[0])])
    # 否则返回t1的副本
    return t1.copy()


# ========== 绘图 ==========
def plot_gp_vs_target(best_ind, gen, x_plot, true_func):
    # 尝试计算最佳个体在x_plot上的评价
    try:
        y_gp = [best_ind.evaluate(x) for x in x_plot]
    except:
        # 如果出现异常，则将y_gp设置为全0
        y_gp = [0 for _ in x_plot]

    # 计算true_func在x_plot上的值
    y_true = [true_func(x) for x in x_plot]

    # 清空当前图像
    plt.clf()
    # 绘制true_func在x_plot上的值，并设置标签和颜色
    plt.plot(x_plot, y_true, label='目标函数', color='blue')
    # 绘制best_ind在x_plot上的评价，并设置标签、颜色和线型
    plt.plot(x_plot, y_gp, '--', label=f'第{gen}代GP拟合', color='red')
    # 设置图像标题
    plt.title(f"第 {gen} 代 GP 拟合")
    # 显示图例
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)


# ========== 主进化函数 ==========
def evolve(pop_size=100, generations=1000):
    # 目标函数
    def true_function(x):
        return x ** 3 + 2 * x ** 2 - x + math.sin(x)

    # 生成50个随机x值
    x_vals = [random.uniform(-3, 3) for _ in range(50)]
    # 计算目标函数的y值
    y_target = [true_function(x) for x in x_vals]
    # 生成x轴的值
    x_plot = [x / 50 for x in range(-150, 151)]

    # 生成初始种群
    population = [generate_random_tree(4) for _ in range(pop_size)]

    # 开启实时绘图
    plt.ion()
    plt.figure(figsize=(8, 5))

    # 遍历每一代
    for gen in range(generations):
        # 计算每个个体的适应度
        scored = [(fitness(ind, x_vals, y_target), ind) for ind in population]
        # 按适应度排序
        scored.sort(key=lambda x: x[0])
        # 获取最佳个体
        best_ind = scored[0][1]
        # 打印当前代数、最佳适应度和表达式
        print(f"第 {gen + 1} 代：最佳适应度 {scored[0][0]:.5f}，表达式：{best_ind}")

        # 绘制当前代数和目标函数
        plot_gp_vs_target(best_ind, gen + 1, x_plot, true_function)

        # 选择前20%的个体作为幸存者
        survivors = [ind.copy() for (_, ind) in scored[:pop_size // 5]]
        # 复制幸存者
        new_population = survivors.copy()
        # 生成新的种群
        while len(new_population) < pop_size:
            # 随机选择两个幸存者
            p1, p2 = random.sample(survivors, 2)
            # 交叉
            child = crossover(p1, p2)
            # 变异
            child = mutate(child)
            # 添加到新种群
            new_population.append(child)

        # 更新种群
        population = new_population

    # 关闭实时绘图
    plt.ioff()
    plt.show()

    # 获取最佳适应度和最佳个体
    best_fit, best_ind = min([(fitness(ind, x_vals, y_target), ind) for ind in population], key=lambda x: x[0])
    # 简化表达式
    simplified = best_ind.simplify()
    # 打印原始和简化后的表达式
    print(f"\n最终最佳表达式（原始）：{best_ind}")
    print(f"最终最佳表达式（简化）：{simplified}")
    return simplified


# ========== 运行 ==========
if __name__ == "__main__":
    evolve()
