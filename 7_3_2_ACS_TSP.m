% ��Ⱥϵͳ�㷨��ACS ant colony system TSP�������
clc;clear;close;
tic;
% ==============    ��ʼ������  ================
City=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
]'; %��������
n=size(City,1);     %��������
m=n;                %��������
NC_max=200;         %����������
Alpha=1;            %��Ϣ����Ҫ�̶�����
Beta=5;             %����ʽ����
Rho=0.5;            %��Ϣ������ϵ��
R_best=zeros(NC_max,n);             % �������·����ʼ�� 0
L_best=inf.*ones(NC_max,1);         % �������·�߳��� inf
Tau=ones(n,n);                      % ��Ϣ�ؾ����ʼ�� 1��Tau������Ϣ��
Tabu=zeros(m,n);                    % ���ڴ洢·���ڵ���룬��iֻ���ϣ���j���ڵ�
NC=1;                               % ����������
D=zeros(n,n);                       % ���м��������ʼ��0
for i=1:n           % ����������
    for j=1:n
        D(i,j)=sqrt((City(i,1)-City(j,1))^2+(City(i,2)-City(j,2))^2);
    end
end
Eta=1./D; % �������ӣ�����Խ�̣���������Խ��

% ==============step2:��mֻ��������ŵ���������  ================
while NC<NC_max
    Randpos=randperm(n); %�������n�����е�����----n�����ظ�������
    Tabu(:,1)=(Randpos(1,1:m))'; % ��n������������е�ǰm����Ϊ��1ֻ���ϵ�·��
    NC
    % ==============step3:mֻ���ϰ�ת��  ================
    for j=2:n       % �����ڵ㲻�㣬����n-1����
        for anti_i=1:m
            visited=Tabu(anti_i,1:(j-1)); % �Ѿ����ʹ��Ľڵ�
            P=zeros(1,n-j+1);% ��¼δ���ʽڵ��ѡ�����
            unvisited=1:n;
            unvisited=setdiff(unvisited,visited); % δ���ʽڵ�
            q0=0.5; % ��ֵ
            if rand<=q0 % ���ѡ����һ���ڵ�
                for k=1:length(unvisited)
                    P(k)=(Tau(visited(end),unvisited(k)))^Alpha*(Eta(visited(end),unvisited(k)))^Beta;
                end
                positon=find(P==max(P));            % ѡ���ֵ
                next_to_visit=unvisited(positon(1));  % ���ѡ����һ���ڵ�
            else
                for k=1:length(unvisited)
                    P(k)=(Tau(visited(end),unvisited(k)))^Alpha*(Eta(visited(end),unvisited(k)))^Beta;
                end
                P=P/sum(P); % ���ʹ�һ��
                pcum=cumsum(P); % �ۼƸ���
                select=find(pcum>=rand); % ����ѡ��
                next_to_visit=unvisited(select(1));  % ѡ����һ���ڵ�)
            end
            Tabu(anti_i,j)=next_to_visit; % ��¼��Tabu��
        end
    end
    if NC>=2
        Tabu(1,:)=R_best(NC-1,:); % ����һ������·����Ϊ��һ����ʼ·��
    end
    % ==============step4:����������ϵ�·������  ================
    L=zeros(m,1); % ��¼�������ϵ�·������
    for i=1:m
        R=Tabu(i,:);
        for j=1:n-1
            L(i)=L(i)+D(R(j),R(j+1));
        end
        L(i)=L(i)+D(R(1),R(m)); % ���Ϸ������ľ���
    end
    L_best(NC)=min(L); % ��¼��ǰ�������·��   
    pos=find(L==L_best(NC)); % �ҵ����·����Ӧ�ĳ�������
    R_best(NC,:)=Tabu(pos(1),:); % ��¼���·��

    % ==============step5:������Ϣ�� ����ȫ����Ϣ�ظ��¹��� ================
    Delta_Tau=zeros(n,n); % ��ʼ����Ϣ����������
    for j=1:(n-1)
        % ֻ��ȫ�����ŵ�·����Ӧ�ø�����Ϣ�ز���
        Delta_Tau(R_best(NC,j),R_best(NC,j+1))=Delta_Tau(R_best(NC,j),R_best(NC,j+1))+1./L_best(NC);    % ��Ϣ������
    end
    % ֻ��ȫ�����ŵ�·����Ӧ�ø�����Ϣ�ز���
    Delta_Tau(R_best(NC,n),R_best(NC,1))=Delta_Tau(R_best(NC,n),R_best(NC,1))+1./L_best(NC);    
    Tau=(1-Rho).*Tau+Rho.*Delta_Tau; % ������Ϣ��
    % ==============step6:���ɱ�����  ================
    Tabu=zeros(m,n);
    NC=NC+1;
end

% ==============step7:������  ================
Pos=find(L_best==min(L_best));  % �ҵ����·��
Shortest_route=R_best(Pos(1),:); % ���·��
Shortest_Length=L_best(Pos(1)); % ���·������
DrawRoute(City,Shortest_route); % �������·��
title('TSP�������---���·��');
toc;