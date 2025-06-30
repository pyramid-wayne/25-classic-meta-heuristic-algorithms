% 人工蜂群算法求解TSP问题   ABC:Artificial Bee Colony Algorithm
% 2025.06.30
clc;clear;close;
tic;
% ==============    step1:初始化参数  ================
cityCoord=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
]'; %城市坐标
cityNum=size(cityCoord,1);          %城市数量
D=disMatrix(cityNum,cityCoord);     %距离矩阵
empBeeNum=60;                       %雇佣蜂数量
onlookBeeNum=60;                  %观察蜂数量
colonySize=empBeeNum+onlookBeeNum;  %种群大小
MaxCycle=1000;                        %最大循环次数
Dim=cityNum;                          %目标函数参数维度
Limit=empBeeNum*cityNum;              %控制参数：引领蜂淘汰上限
Colony=zeros(colonySize,cityNum);       %种群初始化
GlobalBest=0;                       %全局最优解
for i=1:colonySize
    Colony(i,:)=randperm(cityNum);  %随机生成初始解
end
Employed=Colony(1:empBeeNum,:);     %取种群一半作为雇佣蜂，另一半作跟随蜂
solutionValue=calculateSolution(empBeeNum,D,cityNum,Employed);  %计算目标值
[GlobalMin,index]=min(solutionValue);   %全局最优解
bestSolution=Employed(index,:);         %初始化为最小目标解
Cycle=1;                                % 循环计数置为1
reapetTime=zeros(1,empBeeNum);           % 控制参数均置为0
% ==============    step2:迭代寻优  ================
while Cycle<MaxCycle
    % ==============    step2.1:引领蜂阶段  ================
    Employed2=Employed;     % 暂存原解
    for i=1:empBeeNum       % 每只引领蜂都做一次邻域变换---可选方式多种
        Param2Change=fix(rand*cityNum)+1;  %随机选择一个参数
        neighbour=fix(rand*empBeeNum)+1;   % 只要是异于i的蜜源均可在1~empBeeNum内随机选择
        while neighbour==i      % 排除自身
            neighbour=fix(rand*empBeeNum)+1;    % 只要是异于i的蜜源均可在1~empBeeNum内随机选择
        end
        tempOrig=Employed2(i,Param2Change);  % 保存原参数
        Employed2(i,Param2Change)=Employed2(neighbour,Param2Change);  % 邻域变换
        posi=find(Employed2(i,:)==Employed2(i,Param2Change));  % 找到变换后参数的位置
        if size(posi,2)~=1
            posi(Param2Change==posi)=[];  % 删除变更的位置参数
            Employed2(i,posi)=tempOrig;  % 恢复缺少的值
        end
    end
    %==============    step3. 计算目标适值并使用贪心策略并保留好的适值  ================
    solutionValue2=calculateSolution(empBeeNum,D,cityNum,Employed2);  % 计算变换后的目标值
    for j=1:empBeeNum       % 贪心策略保留适值
        if solutionValue2(j)<solutionValue(j)
            reapetTime(j)=0;            % 目标如有改进，计数清零
            Employed(j,:)=Employed2(j,:);  % 保留变换后的解---更新解
            solutionValue(j)=solutionValue2(j);  % 保留变换后的目标值---更新目标值
        end
        reapetTime(j)=reapetTime(j)+1;  % 计数加1
    end
    [currentBest,index]=min(solutionValue);   % 当前最优解及其位置
    if currentBest<GlobalMin    % 更新全局最优解
        GlobalMin=currentBest;
        bestSolution=Employed(index,:);
    end
    fiti=1./(1+solutionValue);  % 计算适应度
    NormFit=fiti/sum(fiti);  % 归一化适应度
    Employed2=Employed;     % 回到舞蹈区把蜜源信息通过舞蹈传达给跟随蜂
    i=1;
    t=0;
    % ============ step4:跟随蜂跟随搜索阶段 =============
    while t<onlookBeeNum
        if rand<NormFit(i)  % 按照概率NormFit(i)选择是否跟随
            t=t+1;          % 若选择跟随，跟随蜂做一次邻域变换---可选方式多种
            Param2Change=fix(rand*cityNum)+1;  % 随机选择一个参数
            neighbour=fix(rand*onlookBeeNum)+1;   % 只要是异于i的蜜源均可在1~empBeeNum内随机选择
            while neighbour==i      % 排除自身
                neighbour=fix(rand*onlookBeeNum)+1;    % 只要是异于i的蜜源均可在1~empBeeNum内随机选择
            end
            tempOrig=Employed2(i,Param2Change);  % 保存原参数
            Employed2(i,Param2Change)=Employed(neighbour,Param2Change);  % 邻域变换
            posi=find(Employed2(i,:)==Employed2(i,Param2Change));  % 找到变换后参数的位置
            if size(posi,2)~=1
                posi(Param2Change==posi)=[];  % 删除变更的位置参数
                Employed2(i,posi)=tempOrig;  % 恢复缺少的值
            end
        end
        i=i+1;
        if i==onlookBeeNum+1    % 恢复到开始，继续选择跟随蜂
            i=1;
        end
    end
    % ============step5. 计算目标适值并使用贪心策略并保留好的适值  =============
    solutionValue2=calculateSolution(empBeeNum,D,cityNum,Employed2);  % 计算变换后的目标值
    for j=1:empBeeNum       % 贪心策略保留适值
        if solutionValue2(j)<solutionValue(j)
            reapetTime(j)=0;            % 目标如有改进，计数清零
            Employed(j,:)=Employed2(j,:);  % 保留变换后的解---更新解
            solutionValue(j)=solutionValue2(j);  % 保留变换后的目标值---更新目标值
        end
        reapetTime(j)=reapetTime(j)+1;  % 计数加1
    end
    [currentBest,index]=min(solutionValue2);   % 当前最优解及其位置
    if currentBest<GlobalMin    % 更新全局最优解
        GlobalMin=currentBest;
        bestSolution=Employed(index,:);
    end
    % ============== step6: 雇佣蜂淘汰阶段 =============
    for j=1:empBeeNum
        if reapetTime(j)>Limit  % 判断是否达到上限
            reapetTime(j)=0;  % 达到上限，计数清零
            Employed(j,:)=randperm(cityNum);  % 随机生成新解(转为侦查蜂)
        end
    end
    Cycle=Cycle+1;  % 迭代计数加1
end
GlobalMin
bestSolution
DrawRoute(cityCoord, bestSolution)