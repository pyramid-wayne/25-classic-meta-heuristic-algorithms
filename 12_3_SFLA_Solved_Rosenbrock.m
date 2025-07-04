% ��������㷨���Rosenbrock���� SFLA: Shuffled Frog Leaping Algorithm
clc;clear;close;
CostFunction=@(x)Rosenbrock(x);   % ����Ŀ�꺯��
nVar=7;                         % ��������          
VarSize=[1 nVar];               % �����±귶Χ
VarMin=-2;                      % ����������
VarMax=2;                       % ����������
MaxIt=2000;                     % ����������
nPopMemeplex=10;                % ��������㷨��ÿ�������Ⱥ���ܵ�����
nPopMemeplex=max(nPopMemeplex,nVar+1);      % Nelder-Mead��׼
nMemeplex=5;                                % ��Ⱥ����
nPop=nPopMemeplex*nMemeplex;                % �ܵ��ܵ�����
I=reshape(1:nPop,nMemeplex,[]);          % �����Ⱥ��ÿ�������Ⱥ�ı��
% SFLA ��ʼ��ֵ
fla_params.q=max(round(0.3*nPopMemeplex),2);        % ��������
fla_params.alpha=3;                          % ��Ⱥִ�д���
fla_params.Lmax=5;                           % �ֲ���������
fla_params.sigma=2;                       % ��С�����Ծ����
fla_params.CostFunction=CostFunction;       % Ŀ�꺯��
fla_params.VarMin=VarMin;                 % ����������
fla_params.VarMax=VarMax;                 % ����������
empty_individual.Postion=[];                % �������ܳ�ʼ��ռ�
empty_individual.Cost=[];                  % �������ܳ�ʼֵ�ռ�
pop= repmat(empty_individual,nPop,1);       % ��ʼ����Ⱥ����
for i=1:nPop        % ��Ⱥ����ֵ
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position);
end
pop=SortPopulation(pop);        % ������Ӧ�ȴ�С����---����Ŀ��ֵ��ֵ����
BestSol=pop(1);                 % ��¼���Ž�
BestCost=nan(MaxIt,1);          % ��¼����Ŀ��ֵ---��ʼ�����Ա������ε���������Ŀ��ֵ
gBestVal=Inf;                 % ��ʼ��ȫ������Ŀ��ֵ
gBestSolu=[];                 % ��ʼ��ȫ�����Ž�
for it=1:MaxIt      % SFLA ��ѭ��������ʼ
    fla_params.BestSol=BestSol;        % �������Ž�
    Memeplex=cell(nMemeplex,1);      % ������Ⱥ����
    % ����Ⱥִ�������㷨
    for j=1:nMemeplex       % ��ÿ����Ⱥִ�оֲ��������
        Memeplex{j}=pop(I(j,:),:);  % ��Ⱥ���飬ֱ��ȡ����j����Ⱥ������
        Memeplex{j}=RunFLA(Memeplex{j},fla_params);  % ��ÿ����Ⱥִ�������㷨
        pop(I(j,:),:)=Memeplex{j};  % ����Ⱥ���ݷŻ���Ⱥ
    end
    pop=SortPopulation(pop);        % ������Ӧ�ȴ�С����---����Ŀ��ֵ��ֵ����
    BestSol=pop(1);                 % ��¼���Ž�
    BestCost(it)=BestSol.Cost;      % ��¼����Ŀ��ֵ
    if BestSol.Cost<gBestVal        % ����ȫ�����ֵ
        gBestVal=BestSol.Cost;
        gBestSolu=BestSol.Position;
    end
    disp(['����������',num2str(it),'  BestCost: ',num2str(BestCost(it)),'  Best Solu: ',num2str(BestSol.Position)]);
end
% ������
gBestVal
gBestSolu
figure;
% ���Ŀ��ֵ���ŵ��������½�������
semilogy(BestCost,'Linewidth',1);
xlabel('��������');
ylabel('Ŀ�꺯��ֵ');
title('SFLA���Rosenbrock����');
grid on