% 差分算法求解Rosenbrock函数
% 使用向量差来扰动向量种群，通过交叉、变异、选择操作，迭代优化目标解。结构简单、收敛速度快、鲁棒性强，解决连续函数的优化问题
% 方法：初始种群，选择两个个体向量，差值加权后与第三个向量求和产生新的向量（变异），
% 变异后再与目标向量混合交叉生成试验向量，如果试验向量由于目标向量则替代更新；
% Rosenbrock: max f(x1,x2)=(1-x1)^2+100*(x2-x1^2)^2
clc;clear;close;
% 初始化参数
fname='rosen';          % 用字符串命名函数。rosen函数以最小化为目标
VTR=1.0e-6;             % 收敛精度
popSize=20;             % 种群大小
D=2;                    % 向量维数
XVmin=[-2,-2];           % 变量定义域：下限
XVmax=[2,2];             % 变量定义域：上限
iterMax=200;           % 最大迭代次数
F=0.8;                 % 缩放因子，通常在[0,2]范围内
CR=0.8;               % 交叉概率
strategy=1;              % 交叉策略选项，此处选1
pop=zeros(popSize,D);     % 初始化种群
for i=1:popSize
    pop(i,:)=XVmin+rand(1,D).*(XVmax-XVmin);  % 初始化种群
end
popold=zeros(size(pop));   % 保存当前种群
val=zeros(1,popSize);         % 初始化目标值向量均为0
ibest=1; 
val(1)=feval(fname,pop(ibest,:));  % 计算初始种群目标值第一个
bestval=Inf;            % 设定为最佳目标值
for i=2:popSize     % 计算各个体的目标值并保留最优结果
    val(i)=feval(fname,pop(i,:));
    if val(i)<bestval
        ibest=i;    % 保留最优目标解和目标值
        bestval=val(i);
    end
end
bestmemit=pop(ibest,:);  % 当前最优解
bestvalit=bestval;      % 当前最优值
bestmem=bestmemit;
% 本程序支持五中差分更新策略，故从种群中随机选出5个个体参与更新
pm1=zeros(popSize,D); 
pm2=zeros(popSize,D);
pm3=zeros(popSize,D);
pm4=zeros(popSize,D);
pm5=zeros(popSize,D);
bm=zeros(popSize,D);        % 最佳个体
ui=zeros(popSize,D);        % 更新后的个体
rot=0:1:popSize-1;          % 旋转索引数组大小popsize
rotd=0:1:D-1;               % 旋转索引数组大小D
rt=zeros(popSize);         % 另一个旋转索引数组大小popsize
rtd=zeros(D);             % 指数交叉的旋转索引数组大小D

iter=1;
% 开始迭代 
while (iter<iterMax) || (bestval>VTR) % 当小于最大迭代、精度没有达到要求
    % 为了增加多样性对原种群进行混合旋转操作
    popold=pop;     % 保存原种群
    ind=randperm(4);  % 原种群做混合预处理
    a1=randperm(popSize);
    rt=rem(rot+ind(1),popSize);         % 求余
    a2=a1(rt+1);
    rt=rem(rot+ind(2),popSize);
    a3=a2(rt+1);
    rt=rem(rot+ind(3),popSize);
    a4=a3(rt+1);
    rt=rem(rot+ind(4),popSize);
    a5=a4(rt+1);
    pm1=popold(a1,:);       % 混合旋转后的种群1
    pm2=popold(a2,:);       % 混合旋转后的种群2
    pm3=popold(a3,:);       % 混合旋转后的种群3
    pm4=popold(a4,:);       % 混合旋转后的种群4
    pm5=popold(a5,:);       % 混合旋转后的种群5
    for i=1:popSize
        bm(i,:)=bestmemit;  % 保存最优个体
    end
    mui=rand(popSize,D)<CR;  % 随机产生屏蔽字
    mui=sort(mui');          % 转置
    for i=1:popSize
        n=floor(rand*D);
        if n>0
            rtd=rem(rotd+n,D);    % 求余
            mui(:,i)=mui(rtd+1,i);  % 对每一个个体随机给出屏蔽字
        end
    end
    mui=mui';       % 转置回去
    mpo=mui<0.5;    % 全部翻转
    % ---交叉变异---
    if strategy==1
        ui=pm1+F*(pm2-pm3);  % 差分更新
        ui=popold.*mpo+ui.*mui;  % 变异
    elseif strategy==2
        ui=bm+F*(pm1-pm2);  % 差分更新
        ui=popold.*mpo+ui.*mui;  % 变异
    elseif strategy==3
        ui=pm3+F*(bm-popold)+F*(pm1-pm2);  % 差分更新
        ui=popold.*mpo+ui.*mui;  % 变异
    elseif strategy==4
        ui=bm+F*(pm1-pm2+pm3-pm4);  % 差分更新
        ui=popold.*mpo+ui.*mui;  % 变异
    elseif strategy==5
        ui=pm1+F*(pm2-pm3+pm4-pm5);  % 差分更新
        ui=popold.*mpo+ui.*mui;  % 变异
    end
    % ---选择---
    for i=1:popSize     % 保留个体最佳值
        tempval=feval(fname,ui(i,:));
        if tempval<val(i)
            pop(i,:)=ui(i,:);
            val(i)=tempval;
            if tempval<bestval          % 与最佳目标值比较如果更好予以保留
                bestval=tempval;
                bestmem=ui(i,:);
            end
        end
    end
    bestmemit=bestmem;      %更新全局最佳解
    iter=iter+1;
end

% 输出结果
disp('最优目标函数值：');disp(bestval);
disp('最优解：');disp(bestmem);
%-------------计算目标函数值-----------------
function y=rosen(x)
    y=(1-x(1))^2+100*(x(2)-x(1)^2)^2;
end