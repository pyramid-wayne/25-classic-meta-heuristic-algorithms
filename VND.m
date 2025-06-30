function [dis,solution]=VND(solution,dislist)
    % VND: Variable Neighborhood Descent 算法
    % 选用三种 邻域结构展开搜索
    dis=inf;    k=-1;    lmax=3;    i=1;
    while i<=lmax
        switch(i)
            case(1)
                neiborSolution=neighborhoodOne(solution); %swap 算子
            case(2)
                neiborSolution=neighborhoodTwo(solution); %two_opt_swap 算子
            case(3)
                neiborSolution=neighborhoodThree(solution); %teo_h_opt_swap 算子
        end
        neighborNum=size(neiborSolution,1);
        for j=1:neighborNum
            temp=CalDist(dislist,neiborSolution(j,:));
            if temp<dis
                dis=temp;
                k=j;
            end
        end
        if dis<CalDist(dislist,solution)
            solution=neiborSolution(k,:);
            i=1;
        else
            i=i+1;
        end
    end
end