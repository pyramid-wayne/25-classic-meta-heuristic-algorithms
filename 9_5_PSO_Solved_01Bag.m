% ��ɢ����Ⱥ�㷨���0-1�������� PSO������Ⱥ�㷨  01Bag��0-1��������
% ����ΪV�ı�����n����Ʒ��ÿ����Ʒ���Լ��ļ�ֵv�ͼ�ֵ��ռ�ռ�w��ÿ����Ʒֻ��ѡ��һ�Σ�������ֵ��
clc;clear;close;
popsize=10;             % ����Ⱥ��ģ
ItCycles=100;           % ����������
Dimension=10;           % ÿ�����ӵ�ά��
c1=2;c2=1.8;            % ѧϰ����
w_max=0.9;w_min=0.4;    % ����Ȩ��
v_max=5;v_min=-5;       % �����ٶȷ�Χ
V=300;                   % ��������
capacity=[95,75,23,73,50,22,6,57,89,98];    % ��Ʒ����
price=[89,59,19,43,100,72,44,16,7,64];      % ��Ʒ��ֵ
penality=2;
eps=1e-20;

velocity=v_min+rand(popsize,Dimension)*(v_max-v_min);  % ��ʼ���ٶ�
new_position=zeros(popsize,size(capacity,2));          % ��ʼ��λ��
individual_best=rand(popsize,Dimension)>0.5;           % ��ʼ����������λ��Ϊ�������ַ���
pbest=zeros(popsize,1);                                % ��ʼ������������Ӧ�ȣ�Ϊ0
for k=1:popsize
    pbest=funcKnapsack(individual_best,capacity,price,V,penality,popsize);   % �������������Ӧ��
end
global_best=zeros(1,Dimension);                           % ��ʼ��ȫ������λ��
global_best_fit=eps;                                      % ��ʼ��ȫ��������Ӧ�ȣ�Ϊ0
vsig=zeros(popsize,Dimension);                            % ��ʼ��sigmoid����ֵ
% ===== ���� =====
for gen=1:ItCycles
    w=w_max-(w_max-w_min)*gen/ItCycles;   % ����Ȩ�����Եݼ�
    for k=1:popsize
        velocity(k,:)=w*velocity(k,:)+c1*rand()*(individual_best(k,:)-new_position(k,:))+c2*rand()*(global_best-new_position(k,:));  % �����ٶ�
        for t=1:Dimension   % �������ӷ����ٶȲ�����������
            if velocity(k,Dimension)>v_max
                velocity(k,Dimension)=v_max;
            end
            if velocity(k,Dimension)<v_min
                velocity(k,Dimension)=v_min;
            end
        end
        vsig(k,:)=1./(1+exp(-velocity(k,:)));  % sigmoid����
        for  t=1:Dimension   % �������ӷ����ٶȲ�����������
            if vsig(k,t)>rand()
                new_position(k,t)=1;
            else
                new_position(k,t)=0;
            end
        end
    end

    % ===== ������嵱ǰֵ =====
    new_fitness=funcKnapsack(new_position,capacity,price,V,penality,popsize);  % ������嵱ǰ��Ӧ��
    % ===== ���������ʷĿ��ֵ =====
    old_fitness=funcKnapsack(individual_best,capacity,price,V,penality,popsize);  % ���������ʷ��Ӧ��
    for i=1:popsize     % �������嵱ǰ���Ž������Ŀ��ֵ
        if new_fitness(i)>old_fitness(i)
            individual_best(i,:)=new_position(i,:);
            pbest(i)=new_fitness(i);
        end
    end
    [currentBest,index]=max(new_fitness);  % �ҳ���ǰ����λ��
    if currentBest>global_best_fit
        global_best=individual_best(index,:);   % ����ȫ�����Ž������Ŀ��ֵ
        global_best_fit=currentBest;
    end
end
% ===== ������ =====
disp('���ս����');
disp(['����ֵ��',num2str(global_best_fit)]);
disp(['��Ʒѡ��',num2str(global_best)]);
