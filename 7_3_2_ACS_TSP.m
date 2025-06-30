% 蚁群系统算法：ACS ant colony system TSP问题求解
clc;clear;close;
tic;
% ==============    初始化参数  ================
City=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
]'; %城市坐标
n=size(City,1);     %城市数量
m=n;                %蚂蚁数量
NC_max=200;         %最大迭代次数
Alpha=1;            %信息素重要程度因子
Beta=5;             %启发式因子
Rho=0.5;            %信息素蒸发系数
R_best=zeros(NC_max,n);             % 各代最佳路径初始化 0
L_best=inf.*ones(NC_max,1);         % 各代最佳路线长度 inf
Tau=ones(n,n);                      % 信息素矩阵初始化 1；Tau残留信息素
Tabu=zeros(m,n);                    % 用于存储路径节点编码，第i只蚂蚁，第j个节点
NC=1;                               % 迭代计数器
D=zeros(n,n);                       % 城市间距离矩阵初始化0
for i=1:n           % 计算距离矩阵
    for j=1:n
        D(i,j)=sqrt((City(i,1)-City(j,1))^2+(City(i,2)-City(j,2))^2);
    end
end
Eta=1./D; % 启发因子，距离越短，启发因子越大

% ==============step2:将m只蚂蚁随机放到蚂蚁上面  ================
while NC<NC_max
    Randpos=randperm(n); %随机产生n个城市的排列----n个不重复的整数
    Tabu(:,1)=(Randpos(1,1:m))'; % 将n个城市随机排列的前m个作为第1只蚂蚁的路径
    NC
    % ==============step3:m只蚂蚁按转移  ================
    for j=2:n       % 出发节点不算，共有n-1个，
        for anti_i=1:m
            visited=Tabu(anti_i,1:(j-1)); % 已经访问过的节点
            P=zeros(1,n-j+1);% 记录未访问节点的选择概率
            unvisited=1:n;
            unvisited=setdiff(unvisited,visited); % 未访问节点
            q0=0.5; % 阈值
            if rand<=q0 % 随机选择下一个节点
                for k=1:length(unvisited)
                    P(k)=(Tau(visited(end),unvisited(k)))^Alpha*(Eta(visited(end),unvisited(k)))^Beta;
                end
                positon=find(P==max(P));            % 选最大值
                next_to_visit=unvisited(positon(1));  % 随机选择下一个节点
            else
                for k=1:length(unvisited)
                    P(k)=(Tau(visited(end),unvisited(k)))^Alpha*(Eta(visited(end),unvisited(k)))^Beta;
                end
                P=P/sum(P); % 概率归一化
                pcum=cumsum(P); % 累计概率
                select=find(pcum>=rand); % 概率选择
                next_to_visit=unvisited(select(1));  % 选择下一个节点)
            end
            Tabu(anti_i,j)=next_to_visit; % 记录到Tabu中
        end
    end
    if NC>=2
        Tabu(1,:)=R_best(NC-1,:); % 将上一代最优路径作为下一代初始路径
    end
    % ==============step4:计算各个蚂蚁的路径距离  ================
    L=zeros(m,1); % 记录各个蚂蚁的路径距离
    for i=1:m
        R=Tabu(i,:);
        for j=1:n-1
            L(i)=L(i)+D(R(j),R(j+1));
        end
        L(i)=L(i)+D(R(1),R(m)); % 加上返回起点的距离
    end
    L_best(NC)=min(L); % 记录当前代的最短路径   
    pos=find(L==L_best(NC)); % 找到最短路径对应的城市序列
    R_best(NC,:)=Tabu(pos(1),:); % 记录最短路径

    % ==============step5:更新信息素 采用全局信息素更新规则 ================
    Delta_Tau=zeros(n,n); % 初始化信息素增量矩阵
    for j=1:(n-1)
        % 只在全局最优的路径上应用更新信息素残留
        Delta_Tau(R_best(NC,j),R_best(NC,j+1))=Delta_Tau(R_best(NC,j),R_best(NC,j+1))+1./L_best(NC);    % 信息素增量
    end
    % 只在全局最优的路径上应用更新信息素残留
    Delta_Tau(R_best(NC,n),R_best(NC,1))=Delta_Tau(R_best(NC,n),R_best(NC,1))+1./L_best(NC);    
    Tau=(1-Rho).*Tau+Rho.*Delta_Tau; % 更新信息素
    % ==============step6:禁忌表清零  ================
    Tabu=zeros(m,n);
    NC=NC+1;
end

% ==============step7:输出结果  ================
Pos=find(L_best==min(L_best));  % 找到最佳路径
Shortest_route=R_best(Pos(1),:); % 最佳路径
Shortest_Length=L_best(Pos(1)); % 最佳路径长度
DrawRoute(City,Shortest_route); % 绘制最佳路径
title('TSP问题求解---最短路劲');
toc;