% ��ɢ�����㷨�Ż�BP������ģ�Ͳ����� SS: Scattered Search  BP: Back Propagation  optimize: �Ż�
% ʹ��logsing()������ΪBP�����缤���������С��������MSEΪĿ�꺯��
% 1������㡢1�������㡢1������㣻280x18��������ck_app10.mat ��280x1��ǩ����Y_10app.mat;��������CK_test120.mat
clc;clear;close;
Nvar=41;        % ��������
hb=10;          % ��������������
lb=-10;         % ��������������
b1=6;           % �����������
b2=4;           % �����Խ����
Nind=30;        % ��ʼ��Ⱥ��ģ���������b1+b2=b
fit='fitnes';    % Ŀ�꺯��
% ====== step1 Ӧ�ö������ķ�������Np����ʼ�� =======
Chrom=rand(Nind,Nvar)*(hb-lb)+lb;   % ���ɳ�ʼ��
% �����η��Ĳ����趨
opts.Chi=2;
opts.Delta=0.01;
opts.Gamma=0.5;
opts.Rho=1;
opts.Sigma=0.5;
opts.MaxIt=200;         % ����������
opts.MaxFunEvals=200;   % ��������۴���
opts.TolFun=1e-10;      % �������۾���
opts.TolX=1e-10;        % �������۾���
Xs=[];                  % �����Ӽ���ʼ��
% ======== step2 Ӧ�þֲ������㷨�������Σ��Ľ���ʼ�����  =======
for i=1:Nind
    X=Chrom(i,:);
    [x,y,X,Y]=simplex(fit,X,opts);      % ���õ������㷨: �ֲ�������ά��Լ�������Թ滮
    % x: ���ص���С����
    % y: ���ص���С��ֵ
    % X: �Ż�����������������Xֵ
    % Y: ���غ�������Ӧֵ
    x(end+1)= y;                  % Ŀ��ֵ
    Xs=[Xs;x];                   % �����Ż���Ľ⼯��x1~x41��x42=���һ��Ŀ��ֵ
end
% ======== step3 Ӧ�òο������·��������ο����� =======
refset2=[];             % ����ο���
[refset,reft]=sortrows(Xs,size(Xs,2));   % ����Ŀ��ֵ��С��������
% ����refset1
refset1=refset(1:b1,:);             % ȡ�ø������⣬ǰ4����Ϊrefset1
it=1;
while it<=5
    % �������
    if it==1
        DistR1=pdist2(refset1(:,1:Nvar),refset((b1+1):Nind,1:Nvar),'euclidean');    % ǰb1����֮�����ľ���
    else
        DistR1=pdist2(refset1(:,1:Nvar),refset(:,1:Nvar),'euclidean');    % ǰb1��������һ������
    end
    for i=1:size(DistR1,2)
        MinDR1(i)=min(DistR1(:,i));         % ÿһ�е���Сֵ;��ÿ���⵽���������С����
    end
    [Fxx,lx]=sort(MinDR1,'descend');                % ������С����Ӵ�С����
    solDiverse=Fxx(1:b2);                             % ȡǰb2����С����
    PsoDiverse=lx(1:b2);                              % ȡǰb2����С�����Ӧ������
    if it==1
        refsetR=refset((b1+1):Nind,:);              % ʣ���
    else
        refsetR=refset;                              % ʣ���
    end
    for i=1:b2
        refse=refset(PsoDiverse(i),:);              % ȡ��ǰb2����
        refset2=[refse;refset2];                    % ȡ�ö����Խ⣬��������b2����
    end
    % ======== step4 Ӧ���Ӽ�����������RefSet��������һ���Ӽ� =======
    New_SolR1=[]; 
    for i=1:b1              % ���ø����������һ��ο���
        for j=i+1:b1
            x=refset1(i,1:Nvar);
            y=refset1(j,1:Nvar);
            New=CombineRefset(x,y,1);
            New_SolR1=[New_SolR1;New];             % �����½�
        end
    end
    New_SolR2=[]; 
    for i=1:b2              % ���ö����Խ������һ��ο���
        for j=i+1:b2
            x=refset2(i,1:Nvar);
            y=refset2(j,1:Nvar);
            New=CombineRefset(x,y,2);
            New_SolR2=[New_SolR2;New];             % �����½�
        end
    end
    % refset1 Vs refset2 �ϲ��Ӽ�
    New_SolR3=[];
    for i=1:b1
        for j=1:b2
            x=refset1(i,1:Nvar);
            y=refset2(j,1:Nvar);
            New=CombineRefset(x,y,3);
            New_SolR3=[New_SolR3;New];             % �����¸�������Ͷ����ԵĽ���������µĽ�
        end
    end
    % ======== step5 Ӧ�þֲ������㷨��һ���Ľ��� =======
    New_SolRef1=[];
    for i=1:size(New_SolR1,1)
        X=New_SolR1(i,:);
        [x,y,X,Y]=simplex(fit,X,opts);      % �Ե�һ��ʹ�þֲ��Ż������Ż�һ��
        x(end+1)= y;                  
        New_SolRef1=[New_SolRef1;x];                   
    end

    New_SolRef2=[];
    for i=1:size(New_SolR2,1)
        X=New_SolR2(i,:);
        [x,y,X,Y]=simplex(fit,X,opts);      % �Եڶ���ʹ�þֲ��Ż������Ż�һ��
        x(end+1)= y;                  
        New_SolRef2=[New_SolRef2;x];                   
    end

    New_SolRef3=[];
    for i=1:size(New_SolR3,1)
        X=New_SolR3(i,:);
        [x,y,X,Y]=simplex(fit,X,opts);      % �Ե�����ʹ�þֲ��Ż������Ż�һ��
        x(end+1)= y;                  
        New_SolRef3=[New_SolRef3;x];                   
    end
    % ======== step6 ���Ӽ��ϳ��½⼯��Ӧ�òο������·��������µĲο���RefSet =======
    New_Sol=[New_SolRef1;New_SolRef2;New_SolRef3];  % �ϲ������
    [refset1,refset]=Update_RefSet1(New_Sol,refset1);  % ���²ο���---�ο������·���RefSet
    it=it+1;
end
refset1;        % ���ս�







    
