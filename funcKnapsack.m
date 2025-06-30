function res=funcKnapsack(x,capacity,weight,bag_volume,penality,popsize)
    % ====== 计算目标函数值 ======
    % x: 可行解，二进制串
    % capacity: 物品体积
    % weight: 每个物品的价值
    % bag_volume: 背包容量
    % penality: 惩罚系数
    % popsize: 种群大小

    % 计算个体适应度
    fitness = zeros(1,popsize);
    total_volume=zeros(1,popsize);
    res=zeros(1,popsize);
    for i = 1:popsize
        fitness(i)=sum(x(i,:).*weight); % 总价值
        total_volume(i)=sum(x(i,:).*capacity); % 总体积
        if total_volume(i)<=bag_volume
            res(i)=fitness(i);
        else
            res(i)=fitness(i)-penality*(total_volume(i)-bag_volume); % 超出一个体积的惩罚2
        end
    end
    res=res';
end