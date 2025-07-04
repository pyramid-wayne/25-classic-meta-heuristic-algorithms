% �����������㷨���Rosenbrock���� CS: Cuckoo Search
clc;clear;close all;
% ======= step1 ��������    ================
N=20; % ��Ⱥ����
D=5; % ά��
T=1000; % ����������
Xmin=-5; % �����½�
Xmax=5; % �����Ͻ�
Pa=0.20; % ��������
bestValue=inf; % ��ʼ������ֵ
nestPop=rand(N,D)*(Xmax-Xmin)+Xmin; % ��ʼ����Ⱥ--- �������n���񳲵ĳ�ʼλ��
trace= zeros(1,T);              % ��ʼ����¼������Ӧ��ֵ������
% ======= step2 ��ʼѭ������ ================
for t=1:T
    levy_nestPop=levy(nestPop,Xmax,Xmin); % ������ά���в���
    % ��ά���к��滻������λ��
    index=find(fitness(nestPop)>fitness(levy_nestPop));
    nestPop(index,:)=levy_nestPop(index,:);
    % �������Pa����һЩ��
    rand_nestPop=nestPop+rand.*heaviside(rand(N,D)-Pa).*(nestPop(randperm(N),:)-nestPop(randperm(N),:));
    rand_nestPop(find(nestPop>Xmax))=Xmax;
    rand_nestPop(find(nestPop<Xmin))=Xmin;
    % ���ո�����̭�󣬸�����λ��
    index=find(fitness(nestPop)>fitness(rand_nestPop));
    nestPop(index,:)=rand_nestPop(index,:);
    % �������Ž�
    [bestV,index]=min(fitness(nestPop));
    if bestValue>bestV      % ��������ֵ
        bestValue=bestV;
        bestSolution=nestPop(index,:);
    end
    trace(t)=bestV;         % ����ÿ�ε��������Ž�
    clf;
    plot(bestSolution,'h')
    axis([0 5 -1 2]);
    title(['����������',num2str(t),'  BestCost: ',num2str(bestValue)]);  % ,'  Best Solu: ',num2str(bestSolution)
    pause(0.05);
end
x=bestSolution;         % ������
y=bestValue;
figure;
plot(trace);
xlabel('��������');ylabel('������Ӧ��ֵ');
title('�����������㷨���Rosenbrock����');
