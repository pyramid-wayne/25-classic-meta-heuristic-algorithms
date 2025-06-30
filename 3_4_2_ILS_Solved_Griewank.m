% ILS 迭代局部搜索算法求解Griewank函数；Griewank函数定义：f(x)=1-∏(i=1,n)cos(x_i/sqrt(i))+1，其中x_i∈[-600,600];函数具有许多局部最小值
clc;clear;close;
fname=@griewank;
Maxiter=10000;  % 最大迭代次数
Ndim=30;        % 变量个数d
Bound=[-100,100];  % 变量取值范围
iteration=0;  % 迭代次数
Popsize=30;  % 种群规模
rdp=0.7;  % 对局部解进行扰动，扰动概率0.7
numLocalSearch=5;  % 局部搜索次数
numPerturbation=10;  % 局部搜索扰动次数
Lowerbound=zeros(Ndim,Popsize);  % 种群中每个个体对应变量下界
Upperbound=zeros(Ndim,Popsize);  % 种群中每个个体对应变量上界
for i=1:Popsize
    Lowerbound(:,i)=Bound(1);       % 设定变量下限值
    Upperbound(:,i)=Bound(2);       % 设定变量上限值
end
Population=Lowerbound+rand(Ndim,Popsize).*(Upperbound-Lowerbound);  % 初始化种群
for i=1:Popsize
    fvalue(i)=fname(Population(:,i));  % 计算种群中每个个体的目标函数值
end
[fvaluebest,index]=min(fvalue);         % 找出当前种群中最优个体
Populationbest=Population(:,index);     % 最优个体
prefvalue=fvalue;                       % 记录当前最优目标函数值

%%迭代计算的主要部分
while iteration<Maxiter
    iteration=iteration+1;          % 局部优化
    for i=1:numLocalSearch          % 做多次局部搜索
        a=Populationbest-1/10.*(Populationbest-Lowerbound(:,i));  % 选取当前最优个体局部搜索下限
        b=Populationbest+1/10.*(Upperbound(:,i)-Populationbest);  % 选取当前最优个体局部搜索上限
        numPerturbation=10;  % 局部搜索扰动次数
        Population_new=zeros(Ndim,numPerturbation);  % 局部随机搜索的种群
        for j=1:numPerturbation                     % 局部搜索扰动多次
            Population_new(:,j)=Populationbest;
            change=rand(Ndim,1)<rdp;  % 随机生成一个与变量个数相同的0-1向量，用于判断哪些变量需要扰动
            Population_new(change,j)=a(change)+(b(change)-a(change)).*rand(1);  % 对需要扰动的变量进行扰动
            fvalue_new(j)=fname(Population_new(:,j));  % 计算扰动后的目标函数值
        end
        [fval_newbest,index_new]=min(fvalue_new);  % 找出扰动后的最优个体
        if fval_newbest<fvaluebest  % 如果扰动后的最优个体比当前最优个体更优，则更新最优个体
            fvaluebest=fval_newbest;  % 更新最优目标函数值
            Populationbest=Population_new(:,index_new);  % 更新最优个体
        end
    end
end
%%输出结果
disp(['最优目标函数值为：',num2str(fvaluebest)]);
disp('最优变量取值为：');Populationbest
figure;
plot(prefvalue,'r');
hold on;
plot(fvaluebest,'bo');
xlabel('迭代次数');
ylabel('目标函数值');
