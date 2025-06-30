function newpop=SeleRoulette(pop,Fitness,Popsize)
    % 轮盘赌函数：选择下一代种群
    totalFitness=sum(Fitness);          % 计算适应度总和
    pFitvalue=Fitness/totalFitness;     % 计算每个个体的适应度概率
    mfitvalue=cumsum(pFitvalue);        % 计算累积适应度概率
    fitin=1;
    newpop=zeros(size(pop));            % 用于存选出的新种群
    while fitin<=Popsize
        rd=rand;
        for i=1:Popsize
            if rd>mfitvalue(i)      % 找到随机数所在区间
                continue;
            else
                newpop(fitin,:)=pop(i,:);
                fitin=fitin+1;
                break;
            end
        end
    end

end