% 进化策略ES求解二元函数:y=21.5+x1*sin(4*pi*x1)+x2*sin(20*pi*x2)最大值，x1∈[-3.0,12.1],x2∈[-4.1,5.8]
% 种群规模：40；迭代次数：200；
% x1=10.4837,x2=5.2850,maxy=38.8485
clc;clear;close;
% 随机生成的初始种群体由μ=miu个个体组成
miu=40;
x1=15.1*rand(1,miu)-3;                  %生成初始解 15.1*rand(1,miu)-3
x2=1.7*rand(1,miu)+4.1;                 % 4.1 5.8  x2的值定义不符
X=[x1;x2];                              % 初始解X矩阵，一列表示一个可行解
sigma=rand(2,miu);                      % 标准差向量 σ=sigma
MaxIter=600;                            % 最大迭代次数
maxy=0;
for iter=1:MaxIter
    lamda=1;
    while lamda<=7*miu                  % 产生lamda个个体，lamda>miu,此处选择lamda=7*miu
        pos=1+fix(rand(1,2)*(miu-1));   % 随机产生1~miu之间的整数值，指定两个位置
        pa1=X(:,pos(1));                % 提取两个位置的(X,σ)做离散重组
        pa2=X(:,pos(2));
        % X采用离散重组
        if rand()<0.5               % 随机选出x1
            option(1)=pa1(1);  
        else
            option(1)=pa2(1);
        end
        if rand()<0.5               % 随机选出x2
            option(2)=pa1(2);
        else
            option(2)=pa2(2);
        end
        
        % sigma采用中值重组
        sigma1=0.5*(sigma(:,pos(1))+sigma(:,pos(2)));           % σ采用中值重组
        Y=option'+sigma1.*randn(2,1);                           % 计算更新，τ=τ'=1.randn() 产生标准正态分布随机数
        if Y(1)>=-3 && Y(1)<=12.1 && Y(2)>=4.1 && Y(2)<=5.8    % 判断是否在可行域内
            offspring(:,lamda)=Y;                               % 在可行域内，则将更新后的个体加入子代群体
            lamda=lamda+1;
        end
    end
    U=offspring;      % 采用μ、λ策略，得到λ个新解

    % 计算目标解
    for i=1:size(U,2)
        temp=U(:,i);
        x1=temp(1);
        x2=temp(2);
        eva(i)=f2(x1,x2);
    end
    % 从lamda个后代，选出miu个最好解
    [m_eval,I]=sort(eva);           % 对目标函数值进行排序
    I1=I(end-miu+1:end);            % 从7*miu个子代中取出最大的miu个,最好解
    X=U(:,I1);                      % 得到7*miu个解中最好的miu个解
    % 更新最大目标值max y；同时记录x1，x2
    if m_eval(end)>maxy
        maxy=m_eval(end);
        opmx=U(:,end);
    end
    max_y(iter)=maxy;   %最大值
    mean_y(iter)=mean(eva(I1));   %平均值
end
plot(1:MaxIter,max_y,'b',1:MaxIter,mean_y,'r'); 
legend('最大值','平均值');
opmx
maxy

function y=f2(x1,x2)
    % 目标函数
    y=21.5+x1*sin(4*pi*x1)+x2*sin(20*pi*x2);
end