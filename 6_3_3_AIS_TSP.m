% 人工免疫系统算法 AIS: Artificial Immune System TSP: Traveling Salesman Problem
clc;clear;close all;
C=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
]'; %城市坐标
cityNum=size(C,1);      %城市数量
D=zeros(cityNum);       %城市距离矩阵
antiNum=100;             %抗体数量
colNum=10;             %克隆个数
MaxG=2000;             %最大迭代次数
Pc=0.1;                 %交叉概率
solution=zeros(cityNum,antiNum); %初始化抗体解
len=zeros(antiNum,1);% 路径长度
for i=1:cityNum
    for j=1:cityNum
        D(i,j)=sqrt((C(i,1)-C(j,1))^2+(C(i,2)-C(j,2))^2);
    end
end
for i=1:antiNum     % 随机生成初始化种群
    solution(:,i)=randperm(cityNum);
end
for i=1:antiNum         % 计算路径长度
    len(i)=routeLength(D,solution(:,i),cityNum);
end
[solutionValue,Index]=sort(len); % 按路径长度从小到大排序
bestSolution=solution(:,Index(1)); % 最优解---保留初始最佳抗体解
bestValue=solutionValue(1); % 最优路径长度---保留初始最佳抗体
orderSolution=solution(:,Index); % 依序保留初始最佳抗体
% 人工免疫系统算法循环
for gen=1:MaxG
    for i=1:antiNum/2       % 前50%抗体，每个克隆colNum个后完成邻域交换
        %==================克隆操作选择和变异=================
        a=orderSolution(:,i);
        Ca=repmat(a,1,colNum);      % 从抗体种群中选择colNum个好的抗体予以克隆
        for j=1:colNum
            p1=floor(1+cityNum*rand());
            p2=floor(1+cityNum*rand());
            while p1==p2
                p1=floor(1+cityNum*rand());
                p2=floor(1+cityNum*rand());
            end
            tmp=Ca(p1,j);       % 变异操作：两个随机位置2-opt
            Ca(p1,j)=Ca(p2,j);
            Ca(p2,j)=tmp;
        end
        Ca(:,1)=orderSolution(:,i); % 保留初始抗体
        %=================克隆抑制=================
        for j=1:colNum          % 计算新的colNum个抗体目标值
            Calen(j)=routeLength(D,Ca(:,j),cityNum); % 计算克隆抗体路径长度
        end
        [SortCalen,Index]=sort(Calen); % 按路径长度从小到大排序
        SortCa=Ca(:,Index); 
        af(:,i)=SortCa(:,1);        % 目标解，变异结果进行再选择，抑制亲和度低的抗体
        alen(i)=SortCalen(1);       % 目标值，保留亲和度高的变异结果称为克隆抑制
    end
    % =================刷新种群=================
    for i=1:antiNum/2       % 淘汰部分种群并随机产生替代种群
        bf(:,i)=randperm(cityNum); % 随机产生替代抗体
        blen(i)=routeLength(D,bf(:,i),cityNum); % 计算随机抗体路径长度
    end
    % =================免疫种群与新种群合并=================
    solution=[af,bf];
    len=[alen,blen];
    [solutionValue,Index]=sort(len); % 按路径长度从小到大排序
    orderSolution=solution(:,Index); % 依序保留初始最佳抗体
    trace(gen)=solutionValue(1); % 记录每一代的最优路径长度
end
% =================输出最优化结果=================
fBest=orderSolution(:,1);
Bestlen=trace(end);
figure
for i=1:cityNum-1
    plot([C(fBest(i),1),C(fBest(i+1),1)],[C(fBest(i),2),C(fBest(i+1),2)],'ro-');hold on;
end
plot([C(fBest(cityNum),1),C(fBest(1),1)],[C(fBest(cityNum),2),C(fBest(1),2)],'bo-');
title(['最优路径长度：',num2str(Bestlen)]);