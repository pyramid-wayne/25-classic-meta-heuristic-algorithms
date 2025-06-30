% 离散粒子群算法求解0-1背包问题 PSO：粒子群算法  01Bag：0-1背包问题
% 容量为V的背包，n个物品，每个物品有自己的价值v和价值所占空间w，每个物品只能选择一次，求最大价值；
clc;clear;close;
popsize=10;             % 粒子群规模
ItCycles=100;           % 最大迭代次数
Dimension=10;           % 每个粒子的维度
c1=2;c2=1.8;            % 学习因子
w_max=0.9;w_min=0.4;    % 惯性权重
v_max=5;v_min=-5;       % 粒子速度范围
V=300;                   % 背包容量
capacity=[95,75,23,73,50,22,6,57,89,98];    % 物品容量
price=[89,59,19,43,100,72,44,16,7,64];      % 物品价值
penality=2;
eps=1e-20;

velocity=v_min+rand(popsize,Dimension)*(v_max-v_min);  % 初始化速度
new_position=zeros(popsize,size(capacity,2));          % 初始化位置
individual_best=rand(popsize,Dimension)>0.5;           % 初始化个体最优位置为二进制字符串
pbest=zeros(popsize,1);                                % 初始化个体最优适应度，为0
for k=1:popsize
    pbest=funcKnapsack(individual_best,capacity,price,V,penality,popsize);   % 计算个体最优适应度
end
global_best=zeros(1,Dimension);                           % 初始化全局最优位置
global_best_fit=eps;                                      % 初始化全局最优适应度，为0
vsig=zeros(popsize,Dimension);                            % 初始化sigmoid函数值
% ===== 迭代 =====
for gen=1:ItCycles
    w=w_max-(w_max-w_min)*gen/ItCycles;   % 惯性权重线性递减
    for k=1:popsize
        velocity(k,:)=w*velocity(k,:)+c1*rand()*(individual_best(k,:)-new_position(k,:))+c2*rand()*(global_best-new_position(k,:));  % 更新速度
        for t=1:Dimension   % 限制粒子飞行速度不超过上下限
            if velocity(k,Dimension)>v_max
                velocity(k,Dimension)=v_max;
            end
            if velocity(k,Dimension)<v_min
                velocity(k,Dimension)=v_min;
            end
        end
        vsig(k,:)=1./(1+exp(-velocity(k,:)));  % sigmoid函数
        for  t=1:Dimension   % 限制粒子飞行速度不超过上下限
            if vsig(k,t)>rand()
                new_position(k,t)=1;
            else
                new_position(k,t)=0;
            end
        end
    end

    % ===== 计算个体当前值 =====
    new_fitness=funcKnapsack(new_position,capacity,price,V,penality,popsize);  % 计算个体当前适应度
    % ===== 计算个体历史目标值 =====
    old_fitness=funcKnapsack(individual_best,capacity,price,V,penality,popsize);  % 计算个体历史适应度
    for i=1:popsize     % 保留个体当前最优解和最优目标值
        if new_fitness(i)>old_fitness(i)
            individual_best(i,:)=new_position(i,:);
            pbest(i)=new_fitness(i);
        end
    end
    [currentBest,index]=max(new_fitness);  % 找出当前最优位置
    if currentBest>global_best_fit
        global_best=individual_best(index,:);   % 保留全局最优解和最优目标值
        global_best_fit=currentBest;
    end
end
% ===== 输出结果 =====
disp('最终结果：');
disp(['最大价值：',num2str(global_best_fit)]);
disp(['物品选择：',num2str(global_best)]);
