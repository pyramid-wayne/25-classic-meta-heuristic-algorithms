% TS Tabu Search Algorithm for TSP;���������㷨���TSP����
clc;clear;close all;
% ��������
city=[
        8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
        4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55;
    ]';
% ================================��������===============================
cityNum=size(city,1);           % ��������
TLLength=ceil(cityNum^0.5);     % ���ɳ���
candidateNum=2*cityNum;         % �����������������n*(n-1)/2������n=cityNum
maxTimes=100;                    % ����������
% ================================��ʼ��===============================
distanceMatrix=getDistanceMatrix(cityNum,city);  % ��ȡ�������
TL=zeros(cityNum);           % ��ʼ�����ɱ�
beCandsNum=6;               % ����⼯����
bestFitnessValue=inf;     % ��ʼ��������Ӧ��ֵ
initSolution=randperm(cityNum);  % ������ɳ�ʼ��
beCands=ones(beCandsNum,4);  % ��ʼ������⼯����Ԫ�飺������š���������������������򽻻����������б��

bestSolution=initSolution;     % ��¼���Ž�
currentSolution=initSolution;  % ��¼��ǰ��
CandsList=zeros(candidateNum,cityNum);  % ��¼����⼯
currentTime=1;           % ��¼��ǰ��������
F=zeros(1,candidateNum);  % ��¼�������Ӧ��ֵ�������ѡ��
while currentTime<=maxTimes
    A=Neiborhood(cityNum,candidateNum);  %  ����һ�鲻�ظ������򽻻�λ��
    for i=1:candidateNum        % �������������
        CandsList(i,:)=currentSolution;     % ��ǰȫ���������
        CandsList(i,[A(i,1),A(i,2)])=currentSolution([A(i,2),A(i,1)]);  % ������������
        F(i)=calculateDistance(CandsList(i,:),distanceMatrix);  % ������Ӧ��ֵ
    end
    % ��F��С��������
    [value,order]=sort(F);
    for i=1:beCandsNum      % ����beCandsNum�����������
        beCands(i,1)=order(i);  % �������
        beCands(i,2)=value(i);  % ���������
        beCands(i,3)=A(order(i),1);  % ���򽻻����������б��
        beCands(i,4)=A(order(i),2);
    end
    if beCands(1,2)<bestFitnessValue  % �������Ž�---����������
        bestFitnessValue=beCands(1,2);  % ����⼯�н�С��Ŀ��ֵ���ԭ����ֵ
        currentSolution=CandsList(beCands(1,1),:);  % ���µ�ǰ��,����⼯�н�С��Ŀ��ֵ��Ӧ�Ľ����ԭ��ǰ��
        bestSolution=currentSolution;  % �������Ž�
        updateHappen=1;  % ���·��������
        TL=updateTabuList(TL,beCands(1,3),beCands(1,4),cityNum,TLLength);  % ���½��ɱ�
    else    % �����ӽ�
        for i=1:beCandsNum
            if TL(beCands(i,3),beCands(i,4))==0  % ����������в��ڽ��ɱ���
                currentSolution=CandsList(beCands(i,1),:);  % ���µ�ǰ��
                updateHappen=1;     % ���·��������
                TL=updateTabuList(TL,beCands(i,3),beCands(i,4),cityNum,TLLength);  % ���½��ɱ�
                break;
            end
        end
    end
    currentTime=currentTime+1;
    if updateHappen==1  % ������·���
        displayResult(currentTime,bestSolution,bestFitnessValue,cityNum,city);
        updateHappen=0;  % ����δ���������
    end
    pause(0.005);
end

DrawRoute(city,bestSolution);  % �������Ž�·��
bestFitnessValue
        