% 分散搜索算法优化BP神经网络模型参数； SS: Scattered Search  BP: Back Propagation  optimize: 优化
% 使用logsing()函数作为BP神经网络激活函数，以最小化均方差MSE为目标函数
% 1个输入层、1个隐含层、1个输出层；280x18特征矩阵；ck_app10.mat 和280x1标签矩阵Y_10app.mat;测试数据CK_test120.mat
clc;clear;close;
Nvar=41;        % 变量个数
hb=10;          % 变量定义域上限
lb=-10;         % 变量定义域下限
b1=6;           % 高质量解个数
b2=4;           % 多样性解个数
Nind=30;        % 初始解群规模，必须大于b1+b2=b
fit='fitnes';    % 目标函数
% ====== step1 应用多样化的方法生成Np个初始解 =======
Chrom=rand(Nind,Nvar)*(hb-lb)+lb;   % 生成初始解
% 单纯形法的参数设定
opts.Chi=2;
opts.Delta=0.01;
opts.Gamma=0.5;
opts.Rho=1;
opts.Sigma=0.5;
opts.MaxIt=200;         % 最大迭代次数
opts.MaxFunEvals=200;   % 最大函数评价次数
opts.TolFun=1e-10;      % 函数评价精度
opts.TolX=1e-10;        % 变量评价精度
Xs=[];                  % 保存子集初始化
% ======== step2 应用局部搜索算法（单纯形）改进初始试验解  =======
for i=1:Nind
    X=Chrom(i,:);
    [x,y,X,Y]=simplex(fit,X,opts);      % 调用单纯形算法: 局部搜索多维无约束非线性规划
    % x: 返回的最小化解
    % y: 返回的最小化值
    % X: 优化过程中评估的所有X值
    % Y: 返回函数的相应值
    x(end+1)= y;                  % 目标值
    Xs=[Xs;x];                   % 产生优化后的解集，x1~x41，x42=最后一列目标值
end
% ======== step3 应用参考集更新方法产生参考集解 =======
refset2=[];             % 保存参考集
[refset,reft]=sortrows(Xs,size(Xs,2));   % 按照目标值从小到大排序
% 计算refset1
refset1=refset(1:b1,:);             % 取得高质量解，前4个作为refset1
it=1;
while it<=5
    % 计算距离
    if it==1
        DistR1=pdist2(refset1(:,1:Nvar),refset((b1+1):Nind,1:Nvar),'euclidean');    % 前b1个与之后多个的距离
    else
        DistR1=pdist2(refset1(:,1:Nvar),refset(:,1:Nvar),'euclidean');    % 前b1个到任意一个距离
    end
    for i=1:size(DistR1,2)
        MinDR1(i)=min(DistR1(:,i));         % 每一列的最小值;即每个解到其他解的最小距离
    end
    [Fxx,lx]=sort(MinDR1,'descend');                % 按照最小距离从大到小排序
    solDiverse=Fxx(1:b2);                             % 取前b2个最小距离
    PsoDiverse=lx(1:b2);                              % 取前b2个最小距离对应的索引
    if it==1
        refsetR=refset((b1+1):Nind,:);              % 剩余解
    else
        refsetR=refset;                              % 剩余解
    end
    for i=1:b2
        refse=refset(PsoDiverse(i),:);              % 取得前b2个解
        refset2=[refse;refset2];                    % 取得多样性解，距离最大的b2个解
    end
    % ======== step4 应用子集产生方法对RefSet操作产生一组子集 =======
    New_SolR1=[]; 
    for i=1:b1              % 利用高质量解产生一组参考解
        for j=i+1:b1
            x=refset1(i,1:Nvar);
            y=refset1(j,1:Nvar);
            New=CombineRefset(x,y,1);
            New_SolR1=[New_SolR1;New];             % 产生新解
        end
    end
    New_SolR2=[]; 
    for i=1:b2              % 利用多样性解产生另一组参考解
        for j=i+1:b2
            x=refset2(i,1:Nvar);
            y=refset2(j,1:Nvar);
            New=CombineRefset(x,y,2);
            New_SolR2=[New_SolR2;New];             % 产生新解
        end
    end
    % refset1 Vs refset2 合并子集
    New_SolR3=[];
    for i=1:b1
        for j=1:b2
            x=refset1(i,1:Nvar);
            y=refset2(j,1:Nvar);
            New=CombineRefset(x,y,3);
            New_SolR3=[New_SolR3;New];             % 产生新高质量解和多样性的解产生多组新的解
        end
    end
    % ======== step5 应用局部搜索算法进一步改进解 =======
    New_SolRef1=[];
    for i=1:size(New_SolR1,1)
        X=New_SolR1(i,:);
        [x,y,X,Y]=simplex(fit,X,opts);      % 对第一组使用局部优化搜索优化一下
        x(end+1)= y;                  
        New_SolRef1=[New_SolRef1;x];                   
    end

    New_SolRef2=[];
    for i=1:size(New_SolR2,1)
        X=New_SolR2(i,:);
        [x,y,X,Y]=simplex(fit,X,opts);      % 对第二组使用局部优化搜索优化一下
        x(end+1)= y;                  
        New_SolRef2=[New_SolRef2;x];                   
    end

    New_SolRef3=[];
    for i=1:size(New_SolR3,1)
        X=New_SolR3(i,:);
        [x,y,X,Y]=simplex(fit,X,opts);      % 对第三组使用局部优化搜索优化一下
        x(end+1)= y;                  
        New_SolRef3=[New_SolRef3;x];                   
    end
    % ======== step6 从子集合成新解集并应用参考集更新方法产生新的参考集RefSet =======
    New_Sol=[New_SolRef1;New_SolRef2;New_SolRef3];  % 合并三组解
    [refset1,refset]=Update_RefSet1(New_Sol,refset1);  % 更新参考集---参考集更新方法RefSet
    it=it+1;
end
refset1;        % 最终解







    
