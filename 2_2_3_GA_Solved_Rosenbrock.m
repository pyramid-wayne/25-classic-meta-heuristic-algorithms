% 遗传算法求解Rosenbrock函数
% Rosenbrock: max f(x1,x2)=(1-x1)^2+100*(x2-x1^2)^2
% x定义域 [-2.048 2.048]
clc;clear;close;
popsize=200;                                                    %种群大小
MaxIter=1000;                                                   %最大迭代次数
P_C=0.8;                                                        %交叉概率
P_M=0.01;                                                       %变异概率
Length1=10;                                                     %基因第一部分长度
Length2=10;                                                     %基因第二部分长度
ChromLength=Length1+Length2;                                    %基因总长度
IterNum=0;                                                      %迭代次数
Population=GenerateInitialPopulation(ChromLength,popsize);      %生成初始种群
[Fitness,x1,x2]=CalculateFitnessValue(popsize,Length1,Length2,Population);  %计算初始种群的目标函数值
% (1-x1)^2+100*(x2-x1^2)^2
[CurrentBest,BestIndex]=max(Fitness);                           %找到种群中的最大值及其位置
BestIndividual=Population(BestIndex,:);                         %找到种群中的最优个体
BestValue=CurrentBest                                          %设为最佳目标函数值
%%开始循环迭代
while IterNum<MaxIter
    IterNum=IterNum+1;
    [Fitness1,x1_1,x2_1]=CalculateFitnessValue(popsize,Length1,Length2,Population);   % 计算初始种群的目标函数值
    Population=SeleRoulette(Population,Fitness1,popsize);                    % 轮盘赌选择新一代种群
    Population=CrossoverOperator(popsize,Population,ChromLength,P_C);       % 交叉操作
    Population=MutationOperator(popsize,Population,ChromLength,P_M);        % 变异操作
    [Fitness2,x1_2,x2_2]=CalculateFitnessValue(popsize,Length1,Length2,Population);   % 计算新一代染色体
    [CurrentBest,BestIndex]=max(Fitness2);                           % 找到种群中的最大值及其位置
    if CurrentBest>BestValue                                        % 保存最佳结果
        disp('更新解')
        BestValue=CurrentBest                                      % 更新最优目标函数值
        x1=x1_2;
        x2=x2_2;
    end
end
disp('最优解为：');
disp(BestIndividual);
x1,x2
(1-x1)^2+100*(x2-x1^2)^2
