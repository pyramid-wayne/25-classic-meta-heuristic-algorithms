% �˹�����ϵͳ�㷨 AIS: Artificial Immune System TSP: Traveling Salesman Problem
clc;clear;close all;
C=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
]'; %��������
cityNum=size(C,1);      %��������
D=zeros(cityNum);       %���о������
antiNum=100;             %��������
colNum=10;             %��¡����
MaxG=2000;             %����������
Pc=0.1;                 %�������
solution=zeros(cityNum,antiNum); %��ʼ�������
len=zeros(antiNum,1);% ·������
for i=1:cityNum
    for j=1:cityNum
        D(i,j)=sqrt((C(i,1)-C(j,1))^2+(C(i,2)-C(j,2))^2);
    end
end
for i=1:antiNum     % ������ɳ�ʼ����Ⱥ
    solution(:,i)=randperm(cityNum);
end
for i=1:antiNum         % ����·������
    len(i)=routeLength(D,solution(:,i),cityNum);
end
[solutionValue,Index]=sort(len); % ��·�����ȴ�С��������
bestSolution=solution(:,Index(1)); % ���Ž�---������ʼ��ѿ����
bestValue=solutionValue(1); % ����·������---������ʼ��ѿ���
orderSolution=solution(:,Index); % ��������ʼ��ѿ���
% �˹�����ϵͳ�㷨ѭ��
for gen=1:MaxG
    for i=1:antiNum/2       % ǰ50%���壬ÿ����¡colNum����������򽻻�
        %==================��¡����ѡ��ͱ���=================
        a=orderSolution(:,i);
        Ca=repmat(a,1,colNum);      % �ӿ�����Ⱥ��ѡ��colNum���õĿ������Կ�¡
        for j=1:colNum
            p1=floor(1+cityNum*rand());
            p2=floor(1+cityNum*rand());
            while p1==p2
                p1=floor(1+cityNum*rand());
                p2=floor(1+cityNum*rand());
            end
            tmp=Ca(p1,j);       % ����������������λ��2-opt
            Ca(p1,j)=Ca(p2,j);
            Ca(p2,j)=tmp;
        end
        Ca(:,1)=orderSolution(:,i); % ������ʼ����
        %=================��¡����=================
        for j=1:colNum          % �����µ�colNum������Ŀ��ֵ
            Calen(j)=routeLength(D,Ca(:,j),cityNum); % �����¡����·������
        end
        [SortCalen,Index]=sort(Calen); % ��·�����ȴ�С��������
        SortCa=Ca(:,Index); 
        af(:,i)=SortCa(:,1);        % Ŀ��⣬������������ѡ�������׺Ͷȵ͵Ŀ���
        alen(i)=SortCalen(1);       % Ŀ��ֵ�������׺Ͷȸߵı�������Ϊ��¡����
    end
    % =================ˢ����Ⱥ=================
    for i=1:antiNum/2       % ��̭������Ⱥ��������������Ⱥ
        bf(:,i)=randperm(cityNum); % ��������������
        blen(i)=routeLength(D,bf(:,i),cityNum); % �����������·������
    end
    % =================������Ⱥ������Ⱥ�ϲ�=================
    solution=[af,bf];
    len=[alen,blen];
    [solutionValue,Index]=sort(len); % ��·�����ȴ�С��������
    orderSolution=solution(:,Index); % ��������ʼ��ѿ���
    trace(gen)=solutionValue(1); % ��¼ÿһ��������·������
end
% =================������Ż����=================
fBest=orderSolution(:,1);
Bestlen=trace(end);
figure
for i=1:cityNum-1
    plot([C(fBest(i),1),C(fBest(i+1),1)],[C(fBest(i),2),C(fBest(i+1),2)],'ro-');hold on;
end
plot([C(fBest(cityNum),1),C(fBest(1),1)],[C(fBest(cityNum),2),C(fBest(1),2)],'bo-');
title(['����·�����ȣ�',num2str(Bestlen)]);