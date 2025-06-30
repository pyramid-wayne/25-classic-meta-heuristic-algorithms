function [dis,solution]=VND(solution,dislist)
    % VND: Variable Neighborhood Descent �㷨
    % ѡ������ ����ṹչ������
    dis=inf;    k=-1;    lmax=3;    i=1;
    while i<=lmax
        switch(i)
            case(1)
                neiborSolution=neighborhoodOne(solution); %swap ����
            case(2)
                neiborSolution=neighborhoodTwo(solution); %two_opt_swap ����
            case(3)
                neiborSolution=neighborhoodThree(solution); %teo_h_opt_swap ����
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