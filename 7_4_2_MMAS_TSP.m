% 最大最小蚂蚁系统算法 MMAS: maximum-minimum ant system algorithm for TSP
% MMAS算法是一种用于解决旅行商问题(TSP)的启发式算法。它通过模拟蚂蚁在图中寻找最短路径的过程来解决问题。
% MMAS算法通过调整蚂蚁的移动规则和路径更新规则，以实现更好的搜索性能和收敛速度。
% MMAS算法的主要步骤如下：初始化蚂蚁群体、计算蚂蚁的路径长度、更新路径信息素、根据信息素更新蚂蚁的移动规则、更新全局最优解、迭代优化直到满足终止条件。
clc;clear;close;
tic;
% ==============    step1:初始化参数  ================
City=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
]'; %城市坐标
n=size(City,1); %城市数量
m=n; %蚂蚁数量
NC_max=100; %最大迭代次数
alpha=1; %信息素重要程度因子
Beta=5; %启发式因子
Rho=0.5; %信息素挥发因子
R_best=zeros(NC_max,n); %全局最优解
L_best=inf.*ones(NC_max,1); %全局最优解路径长度
Tau=ones(n,n); %信息素矩阵
Tabu=zeros(m,n); %禁忌表,记录蚂蚁走过的城市
NC=1; %迭代次数
Sigma=0.05; %参数，信息素平滑机制参数
D=zeros(n,n); %城市间距离矩阵
eps=1.0e-16; %参数，信息素平滑机制参数
for i=1:n
    for j=1:n
        if i~=j
            D(i,j)=sqrt((City(i,1)-City(j,1))^2+(City(i,2)-City(j,2))^2);
        else
            D(i,j)=eps;
        end
    end
end
Eta=1./D; %启发式信息矩阵

% ==============    step2:迭代寻优,m只蚂蚁放在n个节点上，  ================
while NC<=NC_max
    % ==============    step2.1:随机产生蚂蚁的起点  ================
    RandNode=randperm(n);
    Tabu(:,1)=(RandNode(1:m))';
    % ==============    step3:蚂蚁按照概率搜索下一个节点  ================
    for j=2:n
        for ant_i=1:m                    % 1~m只蚂蚁构造访问节点
            visited=Tabu(ant_i,1:j-1);  % 记录已访问节点
            P=zeros(1,(n-j+1));         % 记录未访问节点的选择概率
            unvisited=1:n;
            unvisited=setdiff(unvisited,visited);   % 仅剩未访问节点
            % 应用状态转移规则ACS公式
            q0=0.5;
            if rand<=q0
                for k=1:length(unvisited)
                    P(k)=(Tau(visited(end),unvisited(k)))^alpha*(Eta(visited(end),unvisited(k)))^Beta;
                end
                position=find(P==max(P));   % 选最大值
                next_to_visit=unvisited(position(1));   % x选定下一节点
            else
                for k=1:length(unvisited)
                    P(k)=(Tau(visited(end),unvisited(k)))^alpha*(Eta(visited(end),unvisited(k)))^Beta;
                end
                P=P/sum(P);  % 归一化
                pcum=cumsum(P);
                select=find(pcum>rand);
                next_to_visit=unvisited(select(1));   % x随机选择下一节点
            end
            Tabu(ant_i,j)=next_to_visit;  % 记录蚂蚁i访问的下一个节点
        end
    end
    if NC>=2
        Tabu(1,:)=R_best(NC-1,:); % 记录最佳路径，用于结果输出
    end
    % ==============    step4:计算每只蚂蚁的路径长度  ================
    L=zeros(m,1);
    for i=1:m
        R=Tabu(i,:);
        for j=1:n-1
            L(i)=L(i)+D(R(j),R(j+1));
        end
        L(i)=L(i)+D(R(1),R(n));
    end
    L_best(NC)=min(L);      % 记录本轮最短路径长度
    pos=find(L==L_best(NC));    % 记录最短路径长度对应的蚂蚁位置
    R_best(NC,:)=Tabu(pos(1),:);  % 记录最短路径
    [globBest,globPos]=min(L_best); % 记录全局最短路径长度和位置
    gloR_best=Tabu(globPos,:);
    % 求出TauMax,TauMin 信息素界限
    gb_length=min(L_best);
    TauMax=1/(Rho*gb_length);
    pbest=0.05; %参数，信息素局部更新比例
    pbest=power(pbest,1/n);
    TauMin=TauMax*(1-pbest)/((n/2-1)*pbest);
    % ==============    step5:更新信息素，采用MMAS信息素更新规则  ================
    Delta_Tau=zeros(n,n);
    r0=0.5;
    if r0>rand
        for j=1:(n-1)
            % 全局
            Delta_Tau(gloR_best(j),gloR_best(j+1))=Delta_Tau(gloR_best(j),gloR_best(j+1))+1/globBest;
        end
        % 回到出发点
        Delta_Tau(gloR_best(n),gloR_best(1))=Delta_Tau(gloR_best(n),gloR_best(1))+1/globBest;
    else
        for j=1:(n-1)
            Delta_Tau(R_best(NC,j),R_best(NC,j+1))=Delta_Tau(R_best(NC,j),R_best(NC,j+1))+1/L_best(NC);
        end
        Delta_Tau(R_best(NC,n),R_best(NC,1))=Delta_Tau(R_best(NC,n),R_best(NC,1))+1/L_best(NC);
    end
    Tau=(1-Rho).*Tau+Rho*Delta_Tau;     % 考虑信息素挥发因子更新信息素
    % NC
    % 信息素平滑机制
    if NC>4 && L_best(NC)==L_best(NC-3)==L_best(NC-2)==L_best(NC-1)
        for i=1:n
            for j=1:n
                Tau(i,j)=Tau(i,j)+Sigma*(TauMax-Tau(i,j));
            end
        end
    end
    % 限制区间策略，检查信息素是否置于最大最小值之间
    for i=1:n
        for j=1:n
            if Tau(i,j)>TauMax
                Tau(i,j)=TauMax;
            elseif Tau(i,j)<TauMin
                Tau(i,j)=TauMin;
            end
        end
    end
    % ==============    step6:禁忌表清零  ================
    Tabu=zeros(m,n);
    NC=NC+1;
end
% ==============    step7:输出结果  ================
Pos=find(L_best==min(L_best));      % 寻找最优路径
Shortest_Route=R_best(Pos(1),:);        % 最优路径
Shortest_Length=L_best(Pos(1));     % 最短路径长度
DrawRoute(City,Shortest_Route);         % 绘制最优路径
title(['最短路径长度：',num2str(Shortest_Length)]);


