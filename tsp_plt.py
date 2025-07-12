import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 模拟一些城市坐标
np.random.seed(0)
num_cities = 10
cities = np.random.rand(num_cities, 2) * 100  # 10个城市，坐标范围0~100

# 假设有若干路径顺序作为演化的过程
paths = [
    np.random.permutation(num_cities) for _ in range(20)
]

# 创建图形
fig, ax = plt.subplots()
sc = ax.scatter(cities[:, 0], cities[:, 1], c='red')
line, = ax.plot([], [], 'b-', lw=2)

# 设置图形边界
ax.set_xlim(0, 110)
ax.set_ylim(0, 110)
ax.set_title("TSP 动态路径展示")


# 更新每一帧的路径
def update(frame):
    order = paths[frame]
    ordered_cities = cities[order]
    ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])  # 回到起点
    line.set_data(ordered_cities[:, 0], ordered_cities[:, 1])
    ax.set_title(f"TSP 路径 - 第 {frame + 1} 步")
    return line,


# 动画
ani = animation.FuncAnimation(fig, update, frames=len(paths), interval=500, blit=True)

plt.show()
