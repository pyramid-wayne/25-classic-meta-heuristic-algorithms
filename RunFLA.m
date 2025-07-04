function pop=RunFLA(pop,params)
    % ִ�������㷨�ľֲ��������
    q=params.q;                 % ��������
    alpha=params.alpha;         % ��Ⱥ����������
    Lmax=params.Lmax;           % ��Ⱥ����������
    sigma=params.sigma;         % ��Ծ����
    CostFunction=params.CostFunction;     % Ŀ����Ӧ�Ⱥ���
    VarMin=params.VarMin;         % �����½�
    VarMax=params.VarMax;         % �����Ͻ�
    VarSize=size(pop(1).Position);     % ��������g��ģ
    BestSol=params.BestSol;         % ȫ�����Ž�
    nPop=numel(pop);         % ����pop��Ԫ�ظ�������Ⱥ��ģ
    P=2*(nPop+1-(1:nPop))/(nPop*(nPop+1));     % ����һ����������
    LowerBound=pop(1).Position;     % ����һ���½�����
    UpperBound=pop(1).Position;     % ----------����һ���Ͻ�����pop(1).Position
    for i=2:nPop                        % �޶�����ֵ��Χ
        LowerBound=min(LowerBound,pop(i).Position);
        UpperBound=max(UpperBound,pop(i).Position);
    end
    for it=1:Lmax           % �����������Lmax��
        L=RandSample(P,q);  % ����Ⱥ���ѡ��q��������������λ�ã�
        B=pop(L);          % ����Ⱥ�����ѡ��q��������������λ�ã�
        for k=1:alpha       % ��Ⱥ����������alpha---����Ⱥִ�д���
            [B,SortOrder]=SortPopulation(B);  % ����Ӧ��ֵ����---����
            L=L(SortOrder);
            ImprovementStep2=false;     % ����Ⱥ�ڱȽϺ�ĸ��Ʊ�־  flase=����
            Censorship=false;          % ��ȫ�����Ž�ȽϺ�ĸ��Ʊ�־  flase=����
            % ��Ⱥ����ú�������ıȽϴ���
            NewSol1=B(end);
            Step=sigma*rand(VarSize).*(B(1).Position-B(end).Position);  % ʽ12-2-1 ����
            NewSol1.Position=B(end).Position+Step;  % ʽ12-2-2
            % �жϱ����Ƿ��ڶ�������
            if all(NewSol1.Position>=VarMin) && all(NewSol1.Position<=VarMax)
                NewSol1.Cost=CostFunction(NewSol1.Position);  % ������Ӧ��ֵ
                if NewSol1.Cost<B(end).Cost  % �ж��Ƿ����
                    B(end)=NewSol1;  % ����
                else
                    ImprovementStep2=true;  % û�и���,�������
                end
            else
                ImprovementStep2=true;  % �������ڶ�������,�������
            end
            if ImprovementStep2  % ȫ�������������ıȽϴ���
                NewSol2=B(end);
                Step=sigma*rand(VarSize).*(BestSol.Position-B(end).Position);  % ʽ12-2-3
                NewSol2.Position=B(end).Position+Step;  % ʽ12-2-4
                % �жϱ����Ƿ��ڶ�������
                if all(NewSol2.Position>=VarMin) && all(NewSol2.Position<=VarMax)
                    NewSol2.Cost=CostFunction(NewSol2.Position);  % ������Ӧ��ֵ
                    if NewSol2.Cost<B(end).Cost  % �ж��Ƿ����
                        B(end)=NewSol2;  % ����
                    else
                        Censorship=true;  % û�и���,�������
                    end
                else
                    Censorship=true;  % �������ڶ�������,�������
                end
            end
            if Censorship  % ȫ�������������ıȽϴ���----����滻�����崦��
                B(end).Position=unifrnd(LowerBound,UpperBound);  % �����������
                B(end).Cost=CostFunction(B(end).Position);  % ������Ӧ��ֵ
            end
        end             % ת��һ������Ⱥ
        pop(L)=B;       % ������Ⱥ
    end