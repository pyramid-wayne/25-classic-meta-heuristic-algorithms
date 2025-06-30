function [Fitness,best_x1,best_x2] = CalculateFitnessValue(Popsize,Length1,Length2,pop)
    % ����Ŀ�꺯��ֵ
    Fitness = zeros(Popsize,1);
    best_x1=0;best_x2=0;
    best_value=-10000;
    for i = 1:Popsize
        temp1=DecodeChromosome(pop,0,Length1,i);
        temp2=DecodeChromosome(pop,Length1,Length2,i);
        x1=4.096*temp1/1023-2.048;
        x2=4.096*temp2/1023-2.048;
        Fitness(i) = 100*(x2-x1^2)^2+(1-x1)^2;
        if Fitness(i)>best_value    % ��ȡ��� best_x1,best_x2
            best_x1=x1;
            best_x2=x2;
            best_value=Fitness(i);
        end
    end
    % ������Ӧ�Ⱥ���ֵ
    % ��Ӧ�Ⱥ���ֵԽ�󣬱�ʾ����Խ����
end