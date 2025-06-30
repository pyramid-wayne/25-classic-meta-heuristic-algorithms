% ILS �����ֲ������㷨���Griewank������Griewank�������壺f(x)=1-��(i=1,n)cos(x_i/sqrt(i))+1������x_i��[-600,600];�����������ֲ���Сֵ
clc;clear;close;
fname=@griewank;
Maxiter=10000;  % ����������
Ndim=30;        % ��������d
Bound=[-100,100];  % ����ȡֵ��Χ
iteration=0;  % ��������
Popsize=30;  % ��Ⱥ��ģ
rdp=0.7;  % �Ծֲ�������Ŷ����Ŷ�����0.7
numLocalSearch=5;  % �ֲ���������
numPerturbation=10;  % �ֲ������Ŷ�����
Lowerbound=zeros(Ndim,Popsize);  % ��Ⱥ��ÿ�������Ӧ�����½�
Upperbound=zeros(Ndim,Popsize);  % ��Ⱥ��ÿ�������Ӧ�����Ͻ�
for i=1:Popsize
    Lowerbound(:,i)=Bound(1);       % �趨��������ֵ
    Upperbound(:,i)=Bound(2);       % �趨��������ֵ
end
Population=Lowerbound+rand(Ndim,Popsize).*(Upperbound-Lowerbound);  % ��ʼ����Ⱥ
for i=1:Popsize
    fvalue(i)=fname(Population(:,i));  % ������Ⱥ��ÿ�������Ŀ�꺯��ֵ
end
[fvaluebest,index]=min(fvalue);         % �ҳ���ǰ��Ⱥ�����Ÿ���
Populationbest=Population(:,index);     % ���Ÿ���
prefvalue=fvalue;                       % ��¼��ǰ����Ŀ�꺯��ֵ

%%�����������Ҫ����
while iteration<Maxiter
    iteration=iteration+1;          % �ֲ��Ż�
    for i=1:numLocalSearch          % ����ξֲ�����
        a=Populationbest-1/10.*(Populationbest-Lowerbound(:,i));  % ѡȡ��ǰ���Ÿ���ֲ���������
        b=Populationbest+1/10.*(Upperbound(:,i)-Populationbest);  % ѡȡ��ǰ���Ÿ���ֲ���������
        numPerturbation=10;  % �ֲ������Ŷ�����
        Population_new=zeros(Ndim,numPerturbation);  % �ֲ������������Ⱥ
        for j=1:numPerturbation                     % �ֲ������Ŷ����
            Population_new(:,j)=Populationbest;
            change=rand(Ndim,1)<rdp;  % �������һ�������������ͬ��0-1�����������ж���Щ������Ҫ�Ŷ�
            Population_new(change,j)=a(change)+(b(change)-a(change)).*rand(1);  % ����Ҫ�Ŷ��ı��������Ŷ�
            fvalue_new(j)=fname(Population_new(:,j));  % �����Ŷ����Ŀ�꺯��ֵ
        end
        [fval_newbest,index_new]=min(fvalue_new);  % �ҳ��Ŷ�������Ÿ���
        if fval_newbest<fvaluebest  % ����Ŷ�������Ÿ���ȵ�ǰ���Ÿ�����ţ���������Ÿ���
            fvaluebest=fval_newbest;  % ��������Ŀ�꺯��ֵ
            Populationbest=Population_new(:,index_new);  % �������Ÿ���
        end
    end
end
%%������
disp(['����Ŀ�꺯��ֵΪ��',num2str(fvaluebest)]);
disp('���ű���ȡֵΪ��');Populationbest
figure;
plot(prefvalue,'r');
hold on;
plot(fvaluebest,'bo');
xlabel('��������');
ylabel('Ŀ�꺯��ֵ');
