function [result] = levy(nestPop,Xmax,Xmin)
    % 使用Levy flight算法优化 产生新解
    [N,D]=size(nestPop);
    % 按照Mantegna法则计算
    beta=1.5;
    sigma_u=(gamma(1+beta)*sin(pi*beta/2)/(beta*gamma((1+beta/2)*2^((beta-1)/2))))^(1/beta);
    sigma_v=1;
    u=normrnd(0,sigma_u,N,D);
    v=normrnd(0,sigma_v,N,D);
    step=u./abs(v).^(1/beta);
    alpha=0.01.*(nestPop(randperm(N),:)-nestPop(randperm(N),:));
    nestPop=nestPop+alpha.*step;
    % 限制变量的上下界
    nestPop(find(nestPop>Xmax))=Xmax;
    nestPop(find(nestPop<Xmin))=Xmin;
    result=nestPop;    
end