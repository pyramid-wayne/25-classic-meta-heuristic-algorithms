function [result] = levy(nestPop,Xmax,Xmin)
    % ʹ��Levy flight�㷨�Ż� �����½�
    [N,D]=size(nestPop);
    % ����Mantegna�������
    beta=1.5;
    sigma_u=(gamma(1+beta)*sin(pi*beta/2)/(beta*gamma((1+beta/2)*2^((beta-1)/2))))^(1/beta);
    sigma_v=1;
    u=normrnd(0,sigma_u,N,D);
    v=normrnd(0,sigma_v,N,D);
    step=u./abs(v).^(1/beta);
    alpha=0.01.*(nestPop(randperm(N),:)-nestPop(randperm(N),:));
    nestPop=nestPop+alpha.*step;
    % ���Ʊ��������½�
    nestPop(find(nestPop>Xmax))=Xmax;
    nestPop(find(nestPop<Xmin))=Xmin;
    result=nestPop;    
end