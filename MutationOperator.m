function Population=MutationOperator(pop,Population,ChromLength,P_M)
    % ���캯��
    for i=1:pop
        p=rand;                                 % �������һ��0-1֮�����       
        point=randperm(ChromLength,1);          % �������һ�������
        if p<=P_M
            Population(i,point)=xor(Population(i,point),1); % 0��1��1��0
        end
    end
end