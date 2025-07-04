function [Xnext,Ynext]=AF_swarm(X,i,visual,step,delta,try_num,lbub,lastY)
    % 聚群觅食过程
    Xi=X(i,:);              % 取出当前解
    D=AF_dist(Xi,X);        % 计算norm(Xi-X)距离
    index=find(D>0& D<visual);      % 除去自身和视距之外的人工群
    nf=length(index);               % 计算满足条件的人工鱼群个数
    if nf>0
        for j=1:size(X,2)
            temp=X(:,j);
            Xc(j)=mean(temp(index));    % 计算满足条件的人工群中心
        end
        Yc=AF_rosenbrock(Xc);           % 计算满足条件的人工鱼的目标值
        Yi=lastY(i);                    % 取出当前解的目标值
        if Yc/nf<delta*Yi               % 判断是否满足觅食条件
            Xnext=Xi+rand*step*(Xc-Xi)/norm(Xc-Xi);     % 当前位置向中心位置移动
            for i=1:length(Xnext)
                if Xnext(i)>lbub(2)
                    Xnext(i)=lbub(2);
                end
                if Xnext(i)<lbub(1)
                    Xnext(i)=lbub(1);
                end
            end
            Ynext=AF_rosenbrock(Xnext);  
        else
            [Xnext,Ynext]=AF_prey(Xi,i,visual,step,try_num,lbub,lastY);     % 常规
        end
    else
        [Xnext,Ynext]=AF_prey(Xi,i,visual,step,try_num,lbub,lastY);         % 常规
    end
end