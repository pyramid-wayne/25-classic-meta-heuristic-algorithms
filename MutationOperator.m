function Population=MutationOperator(pop,Population,ChromLength,P_M)
    % 变异函数
    for i=1:pop
        p=rand;                                 % 随机生成一个0-1之间的数       
        point=randperm(ChromLength,1);          % 随机生成一个变异点
        if p<=P_M
            Population(i,point)=xor(Population(i,point),1); % 0变1，1变0
        end
    end
end