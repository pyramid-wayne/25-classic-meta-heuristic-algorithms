function newpop=CrossoverOperator(Popsize,pop,ChromLength,P_C)
    % 交叉函数，两点中间交叉
    newpop1=zeros(Popsize/2,ChromLength);   % 初始化新种群控件
    newpop2=zeros(Popsize/2,ChromLength);
    newpop=[];
    for i=1:Popsize/2
        point=1+randperm(ChromLength-1,2);   % 随机选择两个交叉点
        while point(1)==point(2)
            point=randperm(ChromLength-1,2);
        end
        if point(1)>point(2)    % 位置调整
            temp=point(1);
            point(1)=point(2);
            point(2)=temp;
        end
        temp1=pop(i,:);     % 取出两个原始个体，第i个与第Popsize/2+i个
        temp2=pop(Popsize/2+i,:);
        p=rand;
        if p<P_C
            part1=temp1(point(1):point(2));     % 取出第一个交叉部分
            part2=temp2(point(1):point(2));     % 取出第二个交叉部分
            newpop1(i,:)=[temp1(1:point(1)-1),part2,temp1(point(2)+1:end)];   % 交叉生成新个体
            newpop2(i,:)=[temp2(1:point(1)-1),part1,temp2(point(2)+1:end)];
        else
            newpop1(i,:)=temp1;
            newpop2(i,:)=temp2;
        end
    end
    newpop=[newpop;newpop1;newpop2];
end