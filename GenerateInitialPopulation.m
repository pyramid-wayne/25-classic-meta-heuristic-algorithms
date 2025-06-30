function pop = GenerateInitialPopulation(ChromLength, Popsize)
    % 随机产生初始种群函数
    pop=zeros(Popsize,ChromLength);
    for i=1:Popsize
        for j=1:ChromLength
            if rand<0.5
                pop(i,j)=0;
            else
                pop(i,j)=1;
            end
        end
    end
end