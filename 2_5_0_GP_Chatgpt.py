import random
import math
import operator
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'  # 黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正负号正常显示
# ========== 操作符定义 ==========
ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
}


# ========== 表达式树定义 ==========
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value  # 操作符、变量或常数
        self.left = left
        self.right = right

    def evaluate(self, x):
        if self.value in ops:
            return ops[self.value](self.left.evaluate(x), self.right.evaluate(x))
        elif self.value == 'x':
            return x
        else:
            return float(self.value)

    def __str__(self):
        if self.value in ops:
            return f"({str(self.left)} {self.value} {str(self.right)})"
        else:
            return str(self.value)

    def copy(self):
        if self.value in ops:
            return Node(self.value, self.left.copy(), self.right.copy())
        else:
            return Node(self.value)

    def simplify(self):
        # 终端返回自身
        if self.value not in ops:
            return Node(self.value)

        # 简化左右子树
        left = self.left.simplify()
        right = self.right.simplify()

        # 两边都是常数 → 直接计算
        if left.value not in ops and right.value not in ops:
            try:
                result = ops[self.value](float(left.value), float(right.value))
                return Node(str(result))
            except Exception:
                pass

        # 规则化简
        if self.value == '+':
            if left.value == '0': return right
            if right.value == '0': return left
        if self.value == '*':
            if left.value == '0' or right.value == '0': return Node('0')
            if left.value == '1': return right
            if right.value == '1': return left
        if self.value == '-':
            if right.value == '0': return left
        if self.value == '/':
            if right.value == '1': return left

        return Node(self.value, left, right)


# ========== 树生成、进化相关 ==========
def generate_random_tree(depth):
    if depth == 0 or (depth < 3 and random.random() < 0.3):
        return Node(random.choice(['x', str(random.randint(0, 5))]))
    else:
        op = random.choice(list(ops.keys()))
        return Node(op,
                    generate_random_tree(depth - 1),
                    generate_random_tree(depth - 1))


def fitness(ind, x_vals, y_target):
    try:
        y_pred = [ind.evaluate(x) for x in x_vals]
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_target, y_pred)) / len(x_vals)
        return mse
    except:
        return float('inf')


def mutate(tree, max_depth=3):
    if random.random() < 0.1:
        return generate_random_tree(random.randint(1, max_depth))
    elif tree.value in ops:
        return Node(tree.value, mutate(tree.left), mutate(tree.right))
    else:
        return tree.copy()


def crossover(tree1, tree2):
    if random.random() < 0.1:
        return tree2.copy()
    elif tree1.value in ops and tree2.value in ops:
        return Node(tree1.value,
                    crossover(tree1.left, tree2.left),
                    crossover(tree1.right, tree2.right))
    else:
        return tree1.copy()


# ========== 动态绘图 ==========
def plot_gp_vs_target(best_ind, gen, x_plot):
    try:
        y_gp = [best_ind.evaluate(x) for x in x_plot]
    except:
        y_gp = [0 for _ in x_plot]

    y_true = [x ** 2 + x + 1 for x in x_plot]

    plt.clf()
    plt.plot(x_plot, y_true, label=r"目标函数: $y = x^2 + x + 1$", color='blue')  # label=r"$y = x^2 + x + 1$" 目标函数: x² + x + 1
    plt.plot(x_plot, y_gp, '--', label=f'第{gen}代GP拟合', color='red')
    plt.title(f"第 {gen} 代 GP 拟合")
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)


# ========== GP主函数 ==========
def evolve(pop_size=100, generations=50):
    x_vals = [random.uniform(-2, 2) for _ in range(30)]
    y_target = [x ** 2 + x + 1 for x in x_vals]
    x_plot = [x / 50 for x in range(-100, 101)]

    population = [generate_random_tree(3) for _ in range(pop_size)]

    plt.ion()
    plt.figure(figsize=(8, 5))

    for gen in range(generations):
        scored = [(fitness(ind, x_vals, y_target), ind) for ind in population]
        scored.sort(key=lambda x: x[0])
        best_ind = scored[0][1]
        print(f"第 {gen + 1} 代：最佳适应度 {scored[0][0]:.4f}，表达式：{best_ind}")

        plot_gp_vs_target(best_ind, gen + 1, x_plot)

        survivors = [ind.copy() for (_, ind) in scored[:pop_size // 5]]
        new_population = survivors.copy()
        while len(new_population) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    plt.ioff()
    plt.show()

    best_fit, best_ind = min([(fitness(ind, x_vals, y_target), ind) for ind in population], key=lambda x: x[0])
    simplified = best_ind.simplify()
    print(f"\n最终最佳表达式（原始）：{best_ind}")
    print(f"最终最佳表达式（简化）：{simplified}")
    return simplified


# ========== 运行 ==========
if __name__ == "__main__":
    evolve()
