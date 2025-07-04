function ObjVal=fitnes(Chrom)
    % 使用logsig()函数训练人工神经网络模型参数
    % 输入层：1个；隐藏层：1个；输出层：1个
    % 返回期望输出和实际输出之间的误差MSE
    [Nind,Nvar]=size(Chrom);
    load('Y_10app.mat');    % 分类标号
    load('Ck_app10.mat');   % 特征值280*18
    trin=Ck_app;            % 训练集
    trout=Y;            % 训练集分类标号
    inp=size(trin,2);     % 输入的特征数量
    out=size(trout,2);    % 输出类型0/1
    hidden=2;         % 隐藏层数
    for i=1:Nind        % 使用神经网络函数logsig训练
        x=Chrom(i,:);
        iw=reshape(x(1:inp*hidden),hidden,inp);
        b1=reshape(x(inp*hidden+1:inp*hidden+hidden),hidden,1);
        lw=reshape(x(inp*hidden+hidden+1:inp*hidden+hidden+hidden*out),out,hidden);
        b2=reshape(x(inp*hidden+hidden+hidden*out+1:inp*hidden+hidden+hidden*out+out),out,1);
        yc=logsig(logsig(trin*iw'+repmat(b1',size(trin,1),1))*lw'+repmat(b2',size(trin,1),1));
        ObjVal=mse(trout-yc);
    end
end