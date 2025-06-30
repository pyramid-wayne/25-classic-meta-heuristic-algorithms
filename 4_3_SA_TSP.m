% SA 模拟退火算法求解 TSP 问题
clc;clear;close;
cityNum=20; % 城市数量
Coord=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55];
T0=1000;            % 初温
tau=0.95;           % 降温系数
Ts=1;               % 终止温度 
MaxInnerLoop=50;    % 内循环最大迭代次数
neiborNum=cityNum;       %邻域解最大数量
fare=distance(Coord);       % 计算距离对称矩阵
path=randperm(cityNum);     % 随机生成初始解
pathfar=pathfare(fare,path);   % 计算初始解的距离
bestValue=pathfar;      % 初始化最优解
currentbestValue=bestValue;  % 初始化当前最优解
bestPath=path;            % 初始化最优路径
while T0>=Ts                % 达到总之温度结束
    for in =1:MaxInnerLoop % 内循环模拟等温过程
        e0=pathfare(fare,path); % 计算当前解的距离
        NborNum=Neiborhood(cityNum,neiborNum); % 生成邻域解
        %---------
        swapDone=swap(path,neiborNum,NborNum); % 交换两个城市
        e2=pathfare(fare,swapDone); % 计算交换后的距离
        [better,index]=sort(e2); % 按距离从小到大排序
        e1=better(1,1); % 取最小距离
        newpath=swapDone(index(1),:); % 本轮迭代最好解
        if e1<e0            % 目标值更好，无条件接收
            currentbestValue=e1;
            currentbestPath=newpath;
            path=newpath;       % 更新当前解,把当前最好点设为下一轮起始点
            if bestValue>currentbestValue       % 保留全局最好值
                bestValue=currentbestValue;     % 更新最优解
                bestPath=currentbestPath;     % 更新最优路径
            end
        else            % 按照Metropolis准则接受
            pt=min(1,exp((e0-e1)/T0)); % 以一定概率接受
            if pt>rand
                path=newpath; % 接受劣解
                e0=e1;
            end
        end
    end
    T0=T0*tau; % 降温
    displayResult(i,bestPath,bestValue,cityNum,Coord'); % 显示结果
    pause(0.005)
end
bestPath
bestValue


