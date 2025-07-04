% 人工鱼群算法求解Rosenbrock函数：AFSO：Artificial Fish Swarm Optimization 
% Rosenbrock函数：f(x,y)=(a-x)^2+b(y-x^2)^2，a=1，b=100
% 初始位置：x0=[-5,5]，y0=[-5,5]
% 步长：step=0.1
% 群体大小：N=50
clc;clear;close all;
tic;
fishnum=300;            % 群体大小
Maxgen=500;             % 最大迭代次数
try_num=50;             % 尝试次数
visual=3;               % 视野
step=0.1;               % 步长
delta=0.618;            % 拥挤度参数
lb=-5;ub=10;            % 变量上下界
var=5;                  % 变量个数
%  ============ step1:初始化 =============
lbub=[lb,ub];           % 变量范围
X=zeros(fishnum,var);  % 初始化群体位置
for i=1:fishnum
    for j=1:var
        X(i,j)=(ub-lb)*rand+lb;       % 随机初始化群体位置
    end
end
gen=1;
BestY=Inf*ones(1,Maxgen);   % 记录每一代最优适应度
BestX=zeros(Maxgen,var);    % 记录每一代最优位置
besty=Inf;                  % 初始最优质设为很大的一个数
Y=AF_rosenbrock(X);        % 计算目标值
%  ============ step2:迭代寻优 =============
while gen<=Maxgen
    for i=1:fishnum         % 对每只人工鱼
        [Xi1,Yi1]=AF_swarm(X,i,visual,step,delta,try_num,lbub,Y);  % 聚群行为
        [Xi2,Yi2]=AF_follow(X,i,visual,step,delta,try_num,lbub,Y);  % 追尾行为
        if Yi1<Yi2 % 取两种行为中适应度最好的
            X(i,:)=Xi1;
            Y(i)=Yi1;
        else
            X(i,:)=Xi2;
            Y(i)=Yi2;
        end
    end
    [Ymin,index]=min(Y);  % 找到当代最优适应度及下标
    if Ymin<besty           % 更新全局最优适应度及位置
        besty=Ymin;         % 保留当前最优值
        bestx=X(index,:);   % 保留当前最优位置
        BestY(gen)=Ymin;    % 保留历次迭代最优值
        BestX(gen,:)=bestx; % 保留历次迭代最优位置
    else
        BestY(gen)=BestY(gen-1);    % 若结果为发生变化，持续保留上次最优值
        BestX(gen,:)=BestX(gen-1,:);    % 若结果为发生变化，持续保留上次最优位置
    end
    gen=gen+1;
end
%  ============ step3:输出结果 =============
disp(['最优位置：',num2str(bestx)]);
disp(['最优值：',num2str(besty)]);
toc;



