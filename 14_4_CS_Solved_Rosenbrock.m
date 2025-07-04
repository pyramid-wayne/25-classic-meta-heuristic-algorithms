% 布谷鸟搜索算法求解Rosenbrock函数 CS: Cuckoo Search
clc;clear;close all;
% ======= step1 参数设置    ================
N=20; % 种群数量
D=5; % 维度
T=1000; % 最大迭代次数
Xmin=-5; % 变量下界
Xmax=5; % 变量上界
Pa=0.20; % 抛弃概率
bestValue=inf; % 初始化最优值
nestPop=rand(N,D)*(Xmax-Xmin)+Xmin; % 初始化种群--- 随机生成n个鸟巢的初始位置
trace= zeros(1,T);              % 初始化记录最优适应度值的数组
% ======= step2 开始循环迭代 ================
for t=1:T
    levy_nestPop=levy(nestPop,Xmax,Xmin); % 生成莱维飞行步长
    % 莱维飞行后，替换更新鸟巢位置
    index=find(fitness(nestPop)>fitness(levy_nestPop));
    nestPop(index,:)=levy_nestPop(index,:);
    % 随机概率Pa抛弃一些鸟巢
    rand_nestPop=nestPop+rand.*heaviside(rand(N,D)-Pa).*(nestPop(randperm(N),:)-nestPop(randperm(N),:));
    rand_nestPop(find(nestPop>Xmax))=Xmax;
    rand_nestPop(find(nestPop<Xmin))=Xmin;
    % 按照概率淘汰后，更新鸟巢位置
    index=find(fitness(nestPop)>fitness(rand_nestPop));
    nestPop(index,:)=rand_nestPop(index,:);
    % 更新最优解
    [bestV,index]=min(fitness(nestPop));
    if bestValue>bestV      % 更新最优值
        bestValue=bestV;
        bestSolution=nestPop(index,:);
    end
    trace(t)=bestV;         % 保存每次迭代的最优解
    clf;
    plot(bestSolution,'h')
    axis([0 5 -1 2]);
    title(['迭代次数：',num2str(t),'  BestCost: ',num2str(bestValue)]);  % ,'  Best Solu: ',num2str(bestSolution)
    pause(0.05);
end
x=bestSolution;         % 输出结果
y=bestValue;
figure;
plot(trace);
xlabel('迭代次数');ylabel('最优适应度值');
title('布谷鸟搜索算法求解Rosenbrock函数');
