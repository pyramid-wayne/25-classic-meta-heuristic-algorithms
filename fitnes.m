function ObjVal=fitnes(Chrom)
    % ʹ��logsig()����ѵ���˹�������ģ�Ͳ���
    % ����㣺1�������ز㣺1��������㣺1��
    % �������������ʵ�����֮������MSE
    [Nind,Nvar]=size(Chrom);
    load('Y_10app.mat');    % ������
    load('Ck_app10.mat');   % ����ֵ280*18
    trin=Ck_app;            % ѵ����
    trout=Y;            % ѵ����������
    inp=size(trin,2);     % �������������
    out=size(trout,2);    % �������0/1
    hidden=2;         % ���ز���
    for i=1:Nind        % ʹ�������纯��logsigѵ��
        x=Chrom(i,:);
        iw=reshape(x(1:inp*hidden),hidden,inp);
        b1=reshape(x(inp*hidden+1:inp*hidden+hidden),hidden,1);
        lw=reshape(x(inp*hidden+hidden+1:inp*hidden+hidden+hidden*out),out,hidden);
        b2=reshape(x(inp*hidden+hidden+hidden*out+1:inp*hidden+hidden+hidden*out+out),out,1);
        yc=logsig(logsig(trin*iw'+repmat(b1',size(trin,1),1))*lw'+repmat(b2',size(trin,1),1));
        ObjVal=mse(trout-yc);
    end
end