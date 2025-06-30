% TS Tabu Search Algorithm for TSP;禁忌搜索算法解决TSP问题
clc;clear;close all;
% 城市坐标
city=[
        8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
        4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
    ]';
% ================================参数设置===============================
cityNum=size(city,1);           % 城市数量
TLLength=ceil(cityNum^0.5);     % 禁忌长度
candidateNum=2*cityNum;         % 邻域解数量，不大于n*(n-1)/2的整数n=cityNum
maxTimes=100;                    % 最大迭代次数
% ================================初始化===============================
distanceMatrix=getDistanceMatrix(cityNum,city);  % 获取距离矩阵
TL=zeros(cityNum);           % 初始化禁忌表
beCandsNum=6;               % 邻域解集数量
bestFitnessValue=inf;     % 初始化最优适应度值
initSolution=randperm(cityNum);  % 随机生成初始解
beCands=ones(beCandsNum,4);  % 初始化邻域解集，四元组：邻域解标号、邻域解距离和邻域距离和邻域交换的两个城市编号

bestSolution=initSolution;     % 记录最优解
currentSolution=initSolution;  % 记录当前解
CandsList=zeros(candidateNum,cityNum);  % 记录邻域解集
currentTime=1;           % 记录当前迭代次数
F=zeros(1,candidateNum);  % 记录邻域解适应度值，保存候选解
while currentTime<=maxTimes
    A=Neiborhood(cityNum,candidateNum);  %  返回一组不重复的邻域交换位置
    for i=1:candidateNum        % 生成所有邻域解
        CandsList(i,:)=currentSolution;     % 当前全体城市排序
        CandsList(i,[A(i,1),A(i,2)])=currentSolution([A(i,2),A(i,1)]);  % 交换两个城市
        F(i)=calculateDistance(CandsList(i,:),distanceMatrix);  % 计算适应度值
    end
    % 对F从小到大排序
    [value,order]=sort(F);
    for i=1:beCandsNum      % 整理beCandsNum个最优邻域解
        beCands(i,1)=order(i);  % 邻域解标号
        beCands(i,2)=value(i);  % 邻域解距离和
        beCands(i,3)=A(order(i),1);  % 邻域交换的两个城市编号
        beCands(i,4)=A(order(i),2);
    end
    if beCands(1,2)<bestFitnessValue  % 更新最优解---无条件接受
        bestFitnessValue=beCands(1,2);  % 邻域解集中较小的目标值替代原最优值
        currentSolution=CandsList(beCands(1,1),:);  % 更新当前解,邻域解集中较小的目标值对应的解替代原当前解
        bestSolution=currentSolution;  % 更新最优解
        updateHappen=1;  % 更新发生，标记
        TL=updateTabuList(TL,beCands(1,3),beCands(1,4),cityNum,TLLength);  % 更新禁忌表
    else    % 接受劣解
        for i=1:beCandsNum
            if TL(beCands(i,3),beCands(i,4))==0  % 如果两个城市不在禁忌表中
                currentSolution=CandsList(beCands(i,1),:);  % 更新当前解
                updateHappen=1;     % 更新发生，标记
                TL=updateTabuList(TL,beCands(i,3),beCands(i,4),cityNum,TLLength);  % 更新禁忌表
                break;
            end
        end
    end
    currentTime=currentTime+1;
    if updateHappen==1  % 如果更新发生
        displayResult(currentTime,bestSolution,bestFitnessValue,cityNum,city);
        updateHappen=0;  % 更新未发生，标记
    end
    pause(0.005);
end

DrawRoute(city,bestSolution);  % 绘制最优解路径
bestFitnessValue
        