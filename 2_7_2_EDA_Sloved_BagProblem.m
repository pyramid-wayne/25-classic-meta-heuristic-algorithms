% 2.7 分布估计算法
% 基于概率分布模型；采样统计计算、宏观层面来获得搜索空间的信息
clc;clear;close;
iterations=1000;         %迭代次数
popSize=200;             %种群大小
% 每个物件的体积
weight=[382745,799601,909247,729069,467902,44328,34610,698150,...
        823460,903959,853665,551830,610856,670702,488960,951111,...
        323046,446298,931161,313859,496951,264724,224916,169684];
% 每个物件的价值
value=[825594,1677009,1676628,1523970,943972,97426,69666,1296457,...
        1679693,1902996,1844992,1049289,1252836,1319836,953277,2067538,...
        675367,853655,1826027,65731,901489,577242,466257,369262];
weightMax=6404180;     %背包最大承重
learningRate=0.3;        %学习率
maxSpec=20;             %最大刷新次数
dominantNum=popSize*0.1; % 优势群体个数
dim=size(weight,2);         %维度
% 初始化种群
prob=0.5*ones(1,dim); % 初始化概率
Best_Solution=zeros(iterations,dim+1);  % 保存最优解,每次迭代最优解
Species=zeros(popSize,dim); % 保存各个体受概率分布影响产生的解
for I=1:iterations
    flag=0;
    i=1;
    while i<=popSize        % 针对每个个体
        % I,i
        r=rand(1,dim);      % 随机生成一个0-1之间的数为变量个数的随机值数组
        % r<prob
        Species(i,:)=1.*(r<prob);    % 按照概率模型创建样本
        weightSum=sum(Species(i,:).*weight,2); % 计算个体重量
        if flag>=maxSpec
            Species(i,:)=zeros(1,dim); % 多次仍难达到要求，随机生成一个个体
            flag=0;
        elseif weightSum>weightMax      % 赶超上线
            i=i-1;
            flag=flag+1;
        else
            flag=0;
        end
        i=i+1;
    end
    Fitness=zeros(popSize,1); % 适应度
    for i=1:popSize
        Fitness(i)=sum(Species(i,:).*value,2); % 计算新种群适应度
    end
    [Fitness,index]=sort(Fitness); % 适应度排序
    Best_Solution(I,1)=I;       % 第一列是序号，第2列式目标值，其列是目标解
    Best_Solution(I,2)=Fitness(index(popSize)); % 每轮迭代得到的最优解
    for i=3:dim+2
        Best_Solution(I,i)=Species(index(popSize),i-2);
    end
    % 选取种群中优势群体
    domSpec=zeros(dominantNum,dim);     % 创建选取的优势群体<popSize
    for i=1:dominantNum % 取出dominantNum个优势个体为最佳种群，参与概率分布计算
        domSpec(i,:)=Species(index(popSize-dominantNum+i),:);
    end
    % 更新概率模型
    Ones_Number=sum(domSpec); 
    prob=(1-learningRate)*prob+learningRate*Ones_Number/dominantNum;   % PBIL计算公式
end

%%
disp(strcat('最优解为：',num2str(Best_Solution(iterations,2))));    % 输出最优解




