% SA ģ���˻��㷨��� TSP ����
clc;clear;close;
cityNum=20; % ��������
Coord=[
    8.54, 0.77, 17.02,  0.55, 18.47,  0.61, 10.36, 8.39, 4.85, 17.08,  3.38, 9.59, 7.01, 16.62, 10.84, 2.58,  5.02, 5.78, 17.33, 7.43;
    4.15, 2.52,  4.41, 12.03,  0.70, 11.51, 16.24, 4.47, 1.63, 13.80, 11.28, 4.66, 8.82, 12.65,  5.22, 9.67, 16.23, 6.34,  6.51, 0.55];
T0=1000;            % ����
tau=0.95;           % ����ϵ��
Ts=1;               % ��ֹ�¶� 
MaxInnerLoop=50;    % ��ѭ������������
neiborNum=cityNum;       %������������
fare=distance(Coord);       % �������Գƾ���
path=randperm(cityNum);     % ������ɳ�ʼ��
pathfar=pathfare(fare,path);   % �����ʼ��ľ���
bestValue=pathfar;      % ��ʼ�����Ž�
currentbestValue=bestValue;  % ��ʼ����ǰ���Ž�
bestPath=path;            % ��ʼ������·��
while T0>=Ts                % �ﵽ��֮�¶Ƚ���
    for in =1:MaxInnerLoop % ��ѭ��ģ����¹���
        e0=pathfare(fare,path); % ���㵱ǰ��ľ���
        NborNum=Neiborhood(cityNum,neiborNum); % ���������
        %---------
        swapDone=swap(path,neiborNum,NborNum); % ������������
        e2=pathfare(fare,swapDone); % ���㽻����ľ���
        [better,index]=sort(e2); % �������С��������
        e1=better(1,1); % ȡ��С����
        newpath=swapDone(index(1),:); % ���ֵ�����ý�
        if e1<e0            % Ŀ��ֵ���ã�����������
            currentbestValue=e1;
            currentbestPath=newpath;
            path=newpath;       % ���µ�ǰ��,�ѵ�ǰ��õ���Ϊ��һ����ʼ��
            if bestValue>currentbestValue       % ����ȫ�����ֵ
                bestValue=currentbestValue;     % �������Ž�
                bestPath=currentbestPath;     % ��������·��
            end
        else            % ����Metropolis׼�����
            pt=min(1,exp((e0-e1)/T0)); % ��һ�����ʽ���
            if pt>rand
                path=newpath; % �����ӽ�
                e0=e1;
            end
        end
    end
    T0=T0*tau; % ����
    displayResult(i,bestPath,bestValue,cityNum,Coord'); % ��ʾ���
    pause(0.005)
end
bestPath
bestValue


