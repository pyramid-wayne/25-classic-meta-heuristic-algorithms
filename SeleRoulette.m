function newpop=SeleRoulette(pop,Fitness,Popsize)
    % ���̶ĺ�����ѡ����һ����Ⱥ
    totalFitness=sum(Fitness);          % ������Ӧ���ܺ�
    pFitvalue=Fitness/totalFitness;     % ����ÿ���������Ӧ�ȸ���
    mfitvalue=cumsum(pFitvalue);        % �����ۻ���Ӧ�ȸ���
    fitin=1;
    newpop=zeros(size(pop));            % ���ڴ�ѡ��������Ⱥ
    while fitin<=Popsize
        rd=rand;
        for i=1:Popsize
            if rd>mfitvalue(i)      % �ҵ��������������
                continue;
            else
                newpop(fitin,:)=pop(i,:);
                fitin=fitin+1;
                break;
            end
        end
    end

end