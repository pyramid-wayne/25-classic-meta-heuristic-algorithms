function res=funcKnapsack(x,capacity,weight,bag_volume,penality,popsize)
    % ====== ����Ŀ�꺯��ֵ ======
    % x: ���н⣬�����ƴ�
    % capacity: ��Ʒ���
    % weight: ÿ����Ʒ�ļ�ֵ
    % bag_volume: ��������
    % penality: �ͷ�ϵ��
    % popsize: ��Ⱥ��С

    % ���������Ӧ��
    fitness = zeros(1,popsize);
    total_volume=zeros(1,popsize);
    res=zeros(1,popsize);
    for i = 1:popsize
        fitness(i)=sum(x(i,:).*weight); % �ܼ�ֵ
        total_volume(i)=sum(x(i,:).*capacity); % �����
        if total_volume(i)<=bag_volume
            res(i)=fitness(i);
        else
            res(i)=fitness(i)-penality*(total_volume(i)-bag_volume); % ����һ������ĳͷ�2
        end
    end
    res=res';
end