function [Fitness,best_x1,best_x2] = CalculateFitnessValue(Popsize,Length1,Length2,pop)
    % 计算目标函数值
    Fitness = zeros(Popsize,1);
    best_x1=0;best_x2=0;
    best_value=-10000;
    for i = 1:Popsize
        temp1=DecodeChromosome(pop,0,Length1,i);
        temp2=DecodeChromosome(pop,Length1,Length2,i);
        x1=4.096*temp1/1023-2.048;
        x2=4.096*temp2/1023-2.048;
        Fitness(i) = 100*(x2-x1^2)^2+(1-x1)^2;
        if Fitness(i)>best_value    % 获取最佳 best_x1,best_x2
            best_x1=x1;
            best_x2=x2;
            best_value=Fitness(i);
        end
    end
    % 计算适应度函数值
    % 适应度函数值越大，表示个体越优秀
end