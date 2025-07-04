% �˹���Ⱥ�㷨���Rosenbrock������AFSO��Artificial Fish Swarm Optimization 
% Rosenbrock������f(x,y)=(a-x)^2+b(y-x^2)^2��a=1��b=100
% ��ʼλ�ã�x0=[-5,5]��y0=[-5,5]
% ������step=0.1
% Ⱥ���С��N=50
clc;clear;close all;
tic;
fishnum=300;            % Ⱥ���С
Maxgen=500;             % ����������
try_num=50;             % ���Դ���
visual=3;               % ��Ұ
step=0.1;               % ����
delta=0.618;            % ӵ���Ȳ���
lb=-5;ub=10;            % �������½�
var=5;                  % ��������
%  ============ step1:��ʼ�� =============
lbub=[lb,ub];           % ������Χ
X=zeros(fishnum,var);  % ��ʼ��Ⱥ��λ��
for i=1:fishnum
    for j=1:var
        X(i,j)=(ub-lb)*rand+lb;       % �����ʼ��Ⱥ��λ��
    end
end
gen=1;
BestY=Inf*ones(1,Maxgen);   % ��¼ÿһ��������Ӧ��
BestX=zeros(Maxgen,var);    % ��¼ÿһ������λ��
besty=Inf;                  % ��ʼ��������Ϊ�ܴ��һ����
Y=AF_rosenbrock(X);        % ����Ŀ��ֵ
%  ============ step2:����Ѱ�� =============
while gen<=Maxgen
    for i=1:fishnum         % ��ÿֻ�˹���
        [Xi1,Yi1]=AF_swarm(X,i,visual,step,delta,try_num,lbub,Y);  % ��Ⱥ��Ϊ
        [Xi2,Yi2]=AF_follow(X,i,visual,step,delta,try_num,lbub,Y);  % ׷β��Ϊ
        if Yi1<Yi2 % ȡ������Ϊ����Ӧ����õ�
            X(i,:)=Xi1;
            Y(i)=Yi1;
        else
            X(i,:)=Xi2;
            Y(i)=Yi2;
        end
    end
    [Ymin,index]=min(Y);  % �ҵ�����������Ӧ�ȼ��±�
    if Ymin<besty           % ����ȫ��������Ӧ�ȼ�λ��
        besty=Ymin;         % ������ǰ����ֵ
        bestx=X(index,:);   % ������ǰ����λ��
        BestY(gen)=Ymin;    % �������ε�������ֵ
        BestX(gen,:)=bestx; % �������ε�������λ��
    else
        BestY(gen)=BestY(gen-1);    % �����Ϊ�����仯�����������ϴ�����ֵ
        BestX(gen,:)=BestX(gen-1,:);    % �����Ϊ�����仯�����������ϴ�����λ��
    end
    gen=gen+1;
end
%  ============ step3:������ =============
disp(['����λ�ã�',num2str(bestx)]);
disp(['����ֵ��',num2str(besty)]);
toc;



