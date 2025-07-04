function pop=RunFLA(pop,params)
    % 执行蛙跳算法的局部深度搜索
    q=params.q;                 % 父代数量
    alpha=params.alpha;         % 孙群最大迭代次数
    Lmax=params.Lmax;           % 子群最大迭代次数
    sigma=params.sigma;         % 跳跃步长
    CostFunction=params.CostFunction;     % 目标适应度函数
    VarMin=params.VarMin;         % 变量下界
    VarMax=params.VarMax;         % 变量上界
    VarSize=size(pop(1).Position);     % 变量个数g规模
    BestSol=params.BestSol;         % 全局最优解
    nPop=numel(pop);         % 返回pop中元素个数，子群规模
    P=2*(nPop+1-(1:nPop))/(nPop*(nPop+1));     % 产生一个概率向量
    LowerBound=pop(1).Position;     % 产生一个下界向量
    UpperBound=pop(1).Position;     % ----------产生一个上界向量pop(1).Position
    for i=2:nPop                        % 限定变量值范围
        LowerBound=min(LowerBound,pop(i).Position);
        UpperBound=max(UpperBound,pop(i).Position);
    end
    for it=1:Lmax           % 局域深度搜索Lmax次
        L=RandSample(P,q);  % 从子群随机选择q个解样本（整数位置）
        B=pop(L);          % 从子群中随机选择q个解样本（整数位置）
        for k=1:alpha       % 孙群最大迭代次数alpha---子族群执行次数
            [B,SortOrder]=SortPopulation(B);  % 按适应度值排序---升序
            L=L(SortOrder);
            ImprovementStep2=false;     % 与子群内比较后的改善标志  flase=改善
            Censorship=false;          % 与全局最优解比较后的改善标志  flase=改善
            % 孙群内最好和最差个体的比较处理
            NewSol1=B(end);
            Step=sigma*rand(VarSize).*(B(1).Position-B(end).Position);  % 式12-2-1 变形
            NewSol1.Position=B(end).Position+Step;  % 式12-2-2
            % 判断变量是否在定义域内
            if all(NewSol1.Position>=VarMin) && all(NewSol1.Position<=VarMax)
                NewSol1.Cost=CostFunction(NewSol1.Position);  % 计算适应度值
                if NewSol1.Cost<B(end).Cost  % 判断是否改善
                    B(end)=NewSol1;  % 改善
                else
                    ImprovementStep2=true;  % 没有改善,做出标记
                end
            else
                ImprovementStep2=true;  % 变量不在定义域内,做出标记
            end
            if ImprovementStep2  % 全局最好与最差个体的比较处理
                NewSol2=B(end);
                Step=sigma*rand(VarSize).*(BestSol.Position-B(end).Position);  % 式12-2-3
                NewSol2.Position=B(end).Position+Step;  % 式12-2-4
                % 判断变量是否在定义域内
                if all(NewSol2.Position>=VarMin) && all(NewSol2.Position<=VarMax)
                    NewSol2.Cost=CostFunction(NewSol2.Position);  % 计算适应度值
                    if NewSol2.Cost<B(end).Cost  % 判断是否改善
                        B(end)=NewSol2;  % 改善
                    else
                        Censorship=true;  % 没有改善,做出标记
                    end
                else
                    Censorship=true;  % 变量不在定义域内,做出标记
                end
            end
            if Censorship  % 全局最好与最差个体的比较处理----随机替换最差个体处理
                B(end).Position=unifrnd(LowerBound,UpperBound);  % 随机产生个体
                B(end).Cost=CostFunction(B(end).Position);  % 计算适应度值
            end
        end             % 转下一个子族群
        pop(L)=B;       % 更新孙群
    end