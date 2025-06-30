% 连续粒子群算法求解Goldstein-Price函数：PSO Continuous Algorithm for Goldstein-Price Function；PSO：Particle Swarm Optimization
% Goldestein-Price函数：Goldstein-Price Function
% f(x,y) = (1+(x+y+1)^2*(19-14x+3x^2-14y+6xy+3y^2))*(30+(2x-3y)^2*(18-32x+12x^2+48y-36xy+27y^2))
clc;clear;close;
% ======= step1. 初始化参数 =============
popsize=10;                     % 粒子群规模
dimension=2;                    % 粒子维度
C1=2;C2=1.8;                    % 学习因子----加速常数
VMIN=-4;VMAX=4;                 % 速度限制
XMIN=-2;XMAX=2;                 % 位置限制
ItTimes=1000;                   % 迭代次数
GlXpos=zeros(1,dimension);      % 全局最优位置
vmskmin=VMIN*ones(popsize,dimension);  % 速度下限
vmskmax=VMAX*ones(popsize,dimension);  % 速度上限
VRmin=ones(dimension,1)*-10;
VRmax=ones(dimension,1)*10;
VR=[VRmin,VRmax];
posmaskmin=repmat(VR(1:dimension,1)',popsize,1);  % 位置下限
posmaskmax=repmat(VR(1:dimension,2)',popsize,1);  % 位置上限
posmaskmeth=0;
vel=rand(popsize,dimension)*(VMIN-VMIN)+VMIN;  % 初始化个体速度
Xpos=rand(popsize,dimension)*(XMAX-XMIN)+XMIN;  % 初始化个体位置
Xpos=Xpos+vel;

minposmask_throwaway=Xpos<=posmaskmin;      % 位置下限掩码
minposmask_keep=Xpos>posmaskmin;            
maxposmask_throwaway=Xpos>=posmaskmax;      % 位置上限掩码
maxposmask_keep=Xpos<posmaskmax;
Xpos=(minposmask_throwaway.*posmaskmin)+(minposmask_keep.*Xpos); % 位置下限处理
Xpos=(maxposmask_throwaway.*posmaskmax)+(maxposmask_keep.*Xpos); % 位置上限处理

fitness=goldstein_priceFunc(Xpos,popsize);      % 计算适应度---目标函数适值
lBestXpos=Xpos;                                 % 初始化局部最佳位置
LBest=fitness;                                  % 初始化局部最佳适应度
[GBest,pos]=min(fitness);                       % 全局最佳GBest
GlXpos=Xpos(pos,:);                             % 初始化全局最佳位置

for i=1:ItTimes
    % ======= step2. 计算各粒子目标适值 =============
    fitness=goldstein_priceFunc(Xpos,popsize);  % 计算适应度---目标函数适值
    [bestVal,pos]=min(fitness);               % 全局最佳GBest
    if GBest>bestVal        % 更新全局最优位置
        GBest=bestVal;
        GlXpos=Xpos(pos,:);
    end
    for j=1:popsize
        if LBest(j)>fitness(j)   % 更新个体最优位置
            LBest(j)=fitness(j);
            lBestXpos(j,:)=Xpos(j,:);
        end
    end
    % ======= step3. 更新个体速度和位置 =============
    iwt=0.98-0.3*i/ItTimes;   % 加权因子,惯性权重系数迭代次数线性变化式
    kx=rand(popsize,2);     % 产生0-1随机数矩阵ξ
    et=rand(popsize,2);     % 产生0-1随机数矩阵η
    vel=iwt*vel+C1*kx.*(lBestXpos-Xpos)+et.*C2.*(repmat(GlXpos,popsize,1)-Xpos); % 更新速度
    vel=(vel<=vmskmin).*vmskmin+(vel>vmskmin).*vel;   % 速度下限处理
    vel=(vel>=vmskmax).*vmskmax+(vel<vmskmax).*vel;   % 速度上限处理
    Xpos=Xpos+vel;                                      % 更新位置
    minposmask_throwaway=Xpos<=posmaskmin;      % 位置下限掩码
    minposmask_keep=Xpos>posmaskmin;            
    maxposmask_throwaway=Xpos>=posmaskmax;      % 位置上限掩码
    maxposmask_keep=Xpos<posmaskmax;
end
GBest
GlXpos