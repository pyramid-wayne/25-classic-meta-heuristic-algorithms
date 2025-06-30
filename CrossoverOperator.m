function newpop=CrossoverOperator(Popsize,pop,ChromLength,P_C)
    % ���溯���������м佻��
    newpop1=zeros(Popsize/2,ChromLength);   % ��ʼ������Ⱥ�ؼ�
    newpop2=zeros(Popsize/2,ChromLength);
    newpop=[];
    for i=1:Popsize/2
        point=1+randperm(ChromLength-1,2);   % ���ѡ�����������
        while point(1)==point(2)
            point=randperm(ChromLength-1,2);
        end
        if point(1)>point(2)    % λ�õ���
            temp=point(1);
            point(1)=point(2);
            point(2)=temp;
        end
        temp1=pop(i,:);     % ȡ������ԭʼ���壬��i�����Popsize/2+i��
        temp2=pop(Popsize/2+i,:);
        p=rand;
        if p<P_C
            part1=temp1(point(1):point(2));     % ȡ����һ�����沿��
            part2=temp2(point(1):point(2));     % ȡ���ڶ������沿��
            newpop1(i,:)=[temp1(1:point(1)-1),part2,temp1(point(2)+1:end)];   % ���������¸���
            newpop2(i,:)=[temp2(1:point(1)-1),part1,temp2(point(2)+1:end)];
        else
            newpop1(i,:)=temp1;
            newpop2(i,:)=temp2;
        end
    end
    newpop=[newpop;newpop1;newpop2];
end