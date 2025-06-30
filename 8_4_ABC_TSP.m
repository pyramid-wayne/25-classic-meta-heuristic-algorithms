% �˹���Ⱥ�㷨���TSP����   ABC:Artificial Bee Colony Algorithm
% 2025.06.30
clc;clear;close;
tic;
% ==============    step1:��ʼ������  ================
cityCoord=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
]'; %��������
cityNum=size(cityCoord,1);          %��������
D=disMatrix(cityNum,cityCoord);     %�������
empBeeNum=60;                       %��Ӷ������
onlookBeeNum=60;                  %�۲������
colonySize=empBeeNum+onlookBeeNum;  %��Ⱥ��С
MaxCycle=1000;                        %���ѭ������
Dim=cityNum;                          %Ŀ�꺯������ά��
Limit=empBeeNum*cityNum;              %���Ʋ������������̭����
Colony=zeros(colonySize,cityNum);       %��Ⱥ��ʼ��
GlobalBest=0;                       %ȫ�����Ž�
for i=1:colonySize
    Colony(i,:)=randperm(cityNum);  %������ɳ�ʼ��
end
Employed=Colony(1:empBeeNum,:);     %ȡ��Ⱥһ����Ϊ��Ӷ�䣬��һ���������
solutionValue=calculateSolution(empBeeNum,D,cityNum,Employed);  %����Ŀ��ֵ
[GlobalMin,index]=min(solutionValue);   %ȫ�����Ž�
bestSolution=Employed(index,:);         %��ʼ��Ϊ��СĿ���
Cycle=1;                                % ѭ��������Ϊ1
reapetTime=zeros(1,empBeeNum);           % ���Ʋ�������Ϊ0
% ==============    step2:����Ѱ��  ================
while Cycle<MaxCycle
    % ==============    step2.1:�����׶�  ================
    Employed2=Employed;     % �ݴ�ԭ��
    for i=1:empBeeNum       % ÿֻ����䶼��һ������任---��ѡ��ʽ����
        Param2Change=fix(rand*cityNum)+1;  %���ѡ��һ������
        neighbour=fix(rand*empBeeNum)+1;   % ֻҪ������i����Դ������1~empBeeNum�����ѡ��
        while neighbour==i      % �ų�����
            neighbour=fix(rand*empBeeNum)+1;    % ֻҪ������i����Դ������1~empBeeNum�����ѡ��
        end
        tempOrig=Employed2(i,Param2Change);  % ����ԭ����
        Employed2(i,Param2Change)=Employed2(neighbour,Param2Change);  % ����任
        posi=find(Employed2(i,:)==Employed2(i,Param2Change));  % �ҵ��任�������λ��
        if size(posi,2)~=1
            posi(Param2Change==posi)=[];  % ɾ�������λ�ò���
            Employed2(i,posi)=tempOrig;  % �ָ�ȱ�ٵ�ֵ
        end
    end
    %==============    step3. ����Ŀ����ֵ��ʹ��̰�Ĳ��Բ������õ���ֵ  ================
    solutionValue2=calculateSolution(empBeeNum,D,cityNum,Employed2);  % ����任���Ŀ��ֵ
    for j=1:empBeeNum       % ̰�Ĳ��Ա�����ֵ
        if solutionValue2(j)<solutionValue(j)
            reapetTime(j)=0;            % Ŀ�����иĽ�����������
            Employed(j,:)=Employed2(j,:);  % �����任��Ľ�---���½�
            solutionValue(j)=solutionValue2(j);  % �����任���Ŀ��ֵ---����Ŀ��ֵ
        end
        reapetTime(j)=reapetTime(j)+1;  % ������1
    end
    [currentBest,index]=min(solutionValue);   % ��ǰ���Ž⼰��λ��
    if currentBest<GlobalMin    % ����ȫ�����Ž�
        GlobalMin=currentBest;
        bestSolution=Employed(index,:);
    end
    fiti=1./(1+solutionValue);  % ������Ӧ��
    NormFit=fiti/sum(fiti);  % ��һ����Ӧ��
    Employed2=Employed;     % �ص��赸������Դ��Ϣͨ���赸����������
    i=1;
    t=0;
    % ============ step4:�������������׶� =============
    while t<onlookBeeNum
        if rand<NormFit(i)  % ���ո���NormFit(i)ѡ���Ƿ����
            t=t+1;          % ��ѡ����棬�������һ������任---��ѡ��ʽ����
            Param2Change=fix(rand*cityNum)+1;  % ���ѡ��һ������
            neighbour=fix(rand*onlookBeeNum)+1;   % ֻҪ������i����Դ������1~empBeeNum�����ѡ��
            while neighbour==i      % �ų�����
                neighbour=fix(rand*onlookBeeNum)+1;    % ֻҪ������i����Դ������1~empBeeNum�����ѡ��
            end
            tempOrig=Employed2(i,Param2Change);  % ����ԭ����
            Employed2(i,Param2Change)=Employed(neighbour,Param2Change);  % ����任
            posi=find(Employed2(i,:)==Employed2(i,Param2Change));  % �ҵ��任�������λ��
            if size(posi,2)~=1
                posi(Param2Change==posi)=[];  % ɾ�������λ�ò���
                Employed2(i,posi)=tempOrig;  % �ָ�ȱ�ٵ�ֵ
            end
        end
        i=i+1;
        if i==onlookBeeNum+1    % �ָ�����ʼ������ѡ������
            i=1;
        end
    end
    % ============step5. ����Ŀ����ֵ��ʹ��̰�Ĳ��Բ������õ���ֵ  =============
    solutionValue2=calculateSolution(empBeeNum,D,cityNum,Employed2);  % ����任���Ŀ��ֵ
    for j=1:empBeeNum       % ̰�Ĳ��Ա�����ֵ
        if solutionValue2(j)<solutionValue(j)
            reapetTime(j)=0;            % Ŀ�����иĽ�����������
            Employed(j,:)=Employed2(j,:);  % �����任��Ľ�---���½�
            solutionValue(j)=solutionValue2(j);  % �����任���Ŀ��ֵ---����Ŀ��ֵ
        end
        reapetTime(j)=reapetTime(j)+1;  % ������1
    end
    [currentBest,index]=min(solutionValue2);   % ��ǰ���Ž⼰��λ��
    if currentBest<GlobalMin    % ����ȫ�����Ž�
        GlobalMin=currentBest;
        bestSolution=Employed(index,:);
    end
    % ============== step6: ��Ӷ����̭�׶� =============
    for j=1:empBeeNum
        if reapetTime(j)>Limit  % �ж��Ƿ�ﵽ����
            reapetTime(j)=0;  % �ﵽ���ޣ���������
            Employed(j,:)=randperm(cityNum);  % ��������½�(תΪ����)
        end
    end
    Cycle=Cycle+1;  % ����������1
end
GlobalMin
bestSolution
DrawRoute(cityCoord, bestSolution)