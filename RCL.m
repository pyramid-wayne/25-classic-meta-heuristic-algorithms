function [RCL,sequence] = RCL(Xdata,alfa,probSize)
    % ̰���������������RCL
    start=randperm(probSize,1);     % ���ѡ��һ����ʼ��
    sequence=start;               % ��¼����·��
    while length(sequence)<probSize     % ������·������С����Ⱥ��ģʱ
        rand1=rand();           % ����һ�������
        if rand1>alfa           % ̰��ѡ��
            if sequence(end)==1     % �ų��ڵ�Ϊ1��ֵ
                city=1;
            else
                city=sequence(end)-1;  % ѡ��ǰһ���ڵ�
            end
            rank=ranking(Xdata,city,probSize);    % ���ص�ǰ�ڵ��ֵ
            [~,I]=sort(rank(:,1));      % ����
            rank=rank(I,:);             % �����ľ���
            count=0;                    % ѡ����Сֵ
            next_city=rank(1,2);        % ��ʼ����һ���ڵ� 
            while ismember(next_city,sequence)  % �����ڵ�ǰ�����������ų�
                count=count+1;
                next_city=rank(count,2);        % ѡ����һ���ڵ�
            end
            sequence=[sequence,next_city];      % ����һ���ڵ���뵱ǰ������
        else    % �����ת
            next_city=randperm(probSize,1);       % ���ѡ��һ���ڵ�
            while ismember(next_city,sequence)    % �����ڵ�ǰ�����������ų�
                next_city=randperm(probSize,1);   % ���ѡ��һ���ڵ�
            end
            sequence=[sequence,next_city];        % ����һ���ڵ���뵱ǰ������
        end
    end
    sequence=[sequence,sequence(1)];            % �ص�ԭ��
    RCL=distance_calc(Xdata,sequence);          % ����·������        
end