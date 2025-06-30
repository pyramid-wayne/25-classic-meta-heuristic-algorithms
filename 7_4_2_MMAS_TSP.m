% �����С����ϵͳ�㷨 MMAS: maximum-minimum ant system algorithm for TSP
% MMAS�㷨��һ�����ڽ������������(TSP)������ʽ�㷨����ͨ��ģ��������ͼ��Ѱ�����·���Ĺ�����������⡣
% MMAS�㷨ͨ���������ϵ��ƶ������·�����¹�����ʵ�ָ��õ��������ܺ������ٶȡ�
% MMAS�㷨����Ҫ�������£���ʼ������Ⱥ�塢�������ϵ�·�����ȡ�����·����Ϣ�ء�������Ϣ�ظ������ϵ��ƶ����򡢸���ȫ�����Ž⡢�����Ż�ֱ��������ֹ������
clc;clear;close;
tic;
% ==============    step1:��ʼ������  ================
City=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
]'; %��������
n=size(City,1); %��������
m=n; %��������
NC_max=100; %����������
alpha=1; %��Ϣ����Ҫ�̶�����
Beta=5; %����ʽ����
Rho=0.5; %��Ϣ�ػӷ�����
R_best=zeros(NC_max,n); %ȫ�����Ž�
L_best=inf.*ones(NC_max,1); %ȫ�����Ž�·������
Tau=ones(n,n); %��Ϣ�ؾ���
Tabu=zeros(m,n); %���ɱ�,��¼�����߹��ĳ���
NC=1; %��������
Sigma=0.05; %��������Ϣ��ƽ�����Ʋ���
D=zeros(n,n); %���м�������
eps=1.0e-16; %��������Ϣ��ƽ�����Ʋ���
for i=1:n
    for j=1:n
        if i~=j
            D(i,j)=sqrt((City(i,1)-City(j,1))^2+(City(i,2)-City(j,2))^2);
        else
            D(i,j)=eps;
        end
    end
end
Eta=1./D; %����ʽ��Ϣ����

% ==============    step2:����Ѱ��,mֻ���Ϸ���n���ڵ��ϣ�  ================
while NC<=NC_max
    % ==============    step2.1:����������ϵ����  ================
    RandNode=randperm(n);
    Tabu(:,1)=(RandNode(1:m))';
    % ==============    step3:���ϰ��ո���������һ���ڵ�  ================
    for j=2:n
        for ant_i=1:m                    % 1~mֻ���Ϲ�����ʽڵ�
            visited=Tabu(ant_i,1:j-1);  % ��¼�ѷ��ʽڵ�
            P=zeros(1,(n-j+1));         % ��¼δ���ʽڵ��ѡ�����
            unvisited=1:n;
            unvisited=setdiff(unvisited,visited);   % ��ʣδ���ʽڵ�
            % Ӧ��״̬ת�ƹ���ACS��ʽ
            q0=0.5;
            if rand<=q0
                for k=1:length(unvisited)
                    P(k)=(Tau(visited(end),unvisited(k)))^alpha*(Eta(visited(end),unvisited(k)))^Beta;
                end
                position=find(P==max(P));   % ѡ���ֵ
                next_to_visit=unvisited(position(1));   % xѡ����һ�ڵ�
            else
                for k=1:length(unvisited)
                    P(k)=(Tau(visited(end),unvisited(k)))^alpha*(Eta(visited(end),unvisited(k)))^Beta;
                end
                P=P/sum(P);  % ��һ��
                pcum=cumsum(P);
                select=find(pcum>rand);
                next_to_visit=unvisited(select(1));   % x���ѡ����һ�ڵ�
            end
            Tabu(ant_i,j)=next_to_visit;  % ��¼����i���ʵ���һ���ڵ�
        end
    end
    if NC>=2
        Tabu(1,:)=R_best(NC-1,:); % ��¼���·�������ڽ�����
    end
    % ==============    step4:����ÿֻ���ϵ�·������  ================
    L=zeros(m,1);
    for i=1:m
        R=Tabu(i,:);
        for j=1:n-1
            L(i)=L(i)+D(R(j),R(j+1));
        end
        L(i)=L(i)+D(R(1),R(n));
    end
    L_best(NC)=min(L);      % ��¼�������·������
    pos=find(L==L_best(NC));    % ��¼���·�����ȶ�Ӧ������λ��
    R_best(NC,:)=Tabu(pos(1),:);  % ��¼���·��
    [globBest,globPos]=min(L_best); % ��¼ȫ�����·�����Ⱥ�λ��
    gloR_best=Tabu(globPos,:);
    % ���TauMax,TauMin ��Ϣ�ؽ���
    gb_length=min(L_best);
    TauMax=1/(Rho*gb_length);
    pbest=0.05; %��������Ϣ�ؾֲ����±���
    pbest=power(pbest,1/n);
    TauMin=TauMax*(1-pbest)/((n/2-1)*pbest);
    % ==============    step5:������Ϣ�أ�����MMAS��Ϣ�ظ��¹���  ================
    Delta_Tau=zeros(n,n);
    r0=0.5;
    if r0>rand
        for j=1:(n-1)
            % ȫ��
            Delta_Tau(gloR_best(j),gloR_best(j+1))=Delta_Tau(gloR_best(j),gloR_best(j+1))+1/globBest;
        end
        % �ص�������
        Delta_Tau(gloR_best(n),gloR_best(1))=Delta_Tau(gloR_best(n),gloR_best(1))+1/globBest;
    else
        for j=1:(n-1)
            Delta_Tau(R_best(NC,j),R_best(NC,j+1))=Delta_Tau(R_best(NC,j),R_best(NC,j+1))+1/L_best(NC);
        end
        Delta_Tau(R_best(NC,n),R_best(NC,1))=Delta_Tau(R_best(NC,n),R_best(NC,1))+1/L_best(NC);
    end
    Tau=(1-Rho).*Tau+Rho*Delta_Tau;     % ������Ϣ�ػӷ����Ӹ�����Ϣ��
    % NC
    % ��Ϣ��ƽ������
    if NC>4 && L_best(NC)==L_best(NC-3)==L_best(NC-2)==L_best(NC-1)
        for i=1:n
            for j=1:n
                Tau(i,j)=Tau(i,j)+Sigma*(TauMax-Tau(i,j));
            end
        end
    end
    % ����������ԣ������Ϣ���Ƿ����������Сֵ֮��
    for i=1:n
        for j=1:n
            if Tau(i,j)>TauMax
                Tau(i,j)=TauMax;
            elseif Tau(i,j)<TauMin
                Tau(i,j)=TauMin;
            end
        end
    end
    % ==============    step6:���ɱ�����  ================
    Tabu=zeros(m,n);
    NC=NC+1;
end
% ==============    step7:������  ================
Pos=find(L_best==min(L_best));      % Ѱ������·��
Shortest_Route=R_best(Pos(1),:);        % ����·��
Shortest_Length=L_best(Pos(1));     % ���·������
DrawRoute(City,Shortest_Route);         % ��������·��
title(['���·�����ȣ�',num2str(Shortest_Length)]);


