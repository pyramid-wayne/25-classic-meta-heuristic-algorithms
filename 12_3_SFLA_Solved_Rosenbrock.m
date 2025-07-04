% 混合蛙跳算法求解Rosenbrock函数 SFLA: Shuffled Frog Leaping Algorithm
clc;clear;close;
CostFunction=@(x)Rosenbrock(x);   % 定义目标函数
nVar=7;                         % 变量个数          
VarSize=[1 nVar];               % 变量下标范围
VarMin=-2;                      % 变量定义域
VarMax=2;                       % 变量定义域
MaxIt=2000;                     % 最大迭代次数
nPopMemeplex=10;                % 混合蛙跳算法中每个混合蛙群中蛙的数量
nPopMemeplex=max(nPopMemeplex,nVar+1);      % Nelder-Mead标准
nMemeplex=5;                                % 子群数量
nPop=nPopMemeplex*nMemeplex;                % 总的蛙的数量
I=reshape(1:nPop,nMemeplex,[]);          % 混合蛙群中每个混合蛙群的编号
% SFLA 初始赋值
fla_params.q=max(round(0.3*nPopMemeplex),2);        % 父代个数
fla_params.alpha=3;                          % 孙群执行次数
fla_params.Lmax=5;                           % 局部迭代次数
fla_params.sigma=2;                       % 最小最大跳跃步长
fla_params.CostFunction=CostFunction;       % 目标函数
fla_params.VarMin=VarMin;                 % 变量定义域
fla_params.VarMax=VarMax;                 % 变量定义域
empty_individual.Postion=[];                % 个体青蛙初始解空间
empty_individual.Cost=[];                  % 个体青蛙初始值空间
pop= repmat(empty_individual,nPop,1);       % 初始化种群矩阵
for i=1:nPop        % 种群赋初值
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position);
end
pop=SortPopulation(pop);        % 按照适应度大小排序---按照目标值升值排序
BestSol=pop(1);                 % 记录最优解
BestCost=nan(MaxIt,1);          % 记录最优目标值---初始化用以保存历次迭代的最优目标值
gBestVal=Inf;                 % 初始化全局最优目标值
gBestSolu=[];                 % 初始化全局最优解
for it=1:MaxIt      % SFLA 主循环迭代开始
    fla_params.BestSol=BestSol;        % 传递最优解
    Memeplex=cell(nMemeplex,1);      % 方便子群处理
    % 分子群执行蛙跳算法
    for j=1:nMemeplex       % 对每个子群执行局部深度搜索
        Memeplex{j}=pop(I(j,:),:);  % 子群分组，直接取出第j个子群组数据
        Memeplex{j}=RunFLA(Memeplex{j},fla_params);  % 对每个子群执行蛙跳算法
        pop(I(j,:),:)=Memeplex{j};  % 将子群数据放回种群
    end
    pop=SortPopulation(pop);        % 按照适应度大小排序---按照目标值升值排序
    BestSol=pop(1);                 % 记录最优解
    BestCost(it)=BestSol.Cost;      % 记录最优目标值
    if BestSol.Cost<gBestVal        % 保存全局最好值
        gBestVal=BestSol.Cost;
        gBestSolu=BestSol.Position;
    end
    disp(['迭代次数：',num2str(it),'  BestCost: ',num2str(BestCost(it)),'  Best Solu: ',num2str(BestSol.Position)]);
end
% 输出结果
gBestVal
gBestSolu
figure;
% 输出目标值随着迭代次数下降的曲线
semilogy(BestCost,'Linewidth',1);
xlabel('迭代次数');
ylabel('目标函数值');
title('SFLA求解Rosenbrock函数');
grid on