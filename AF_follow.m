function [Xnext,Ynext] = AF_follow(X,i,visual,step,delta,try_num,lbub,lastY)
    % 追尾觅食过程
    Xi=X(i,:);      % 取出当前解
    D=AF_dist(Xi,X);    % 计算norm(Xi-X)距离
    index=find(D>0&D<visual);    % 找出距离在[0,visual]范围内的个体
    nf=length(index);    % 计算距离在[0,visual]范围内的个体个数
    if nf>0
        XX=X(index,:);
        YY=lastY(index);
        [Ymin,min_index]=min(YY);       % 满足条件的人工鱼的最小位置
        Xmin=XX(min_index,:);
        Yi=lastY(i);
        if Ymin/nf<delta*Yi     % 如果该式不成立，转去执行常规觅食行为
            Xnext=Xi+rand*step*(Xmin-Xi)/norm(Xmin-Xi);    % 当前位置就向位置移动一步
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
            [Xnext,Ynext]=AF_prey(X(i,:),i,visual,step,try_num,lbub,lastY);    % 执行常规觅食行为
        end
    else
        [Xnext,Ynext]=AF_prey(X(i,:),i,visual,step,try_num,lbub,lastY);    % 执行常规觅食行为
    end
end