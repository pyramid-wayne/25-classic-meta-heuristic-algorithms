function [Xnext,Ynext]=AF_prey(Xi,ii,visual,step,try_num,lbub,lastY)
    % 常规觅食过程
    Xnext=[];
    Yi=lastY(ii);
    for i=1:try_num     % 允许尝试多次
        Xj=Xi+((2*rand(length(Xi),1)-1)*visual)';
        Yj=AF_rosenbrock(Xj);
        if Yj<Yi        % 判断移动效果
            Xnext=Xi+rand*step*(Xj-Xi)/norm(Xj-Xi); % 接受移动一步
            for j=1:length(Xnext)
                if Xnext(j)>lbub(2)
                    Xnext(j)=lbub(2);
                elseif Xnext(j)<lbub(1)
                    Xnext(j)=lbub(1);
                end
            end
            Xi=Xnext;
            break;
        end
    end
    % 随机行为
    if isempty(Xnext)       % 尝试多次之后未发生变化，最后允许执行一次随机变化
        Xj=Xi+((2*rand(length(Xi),1)-1)*visual)';
        Xnext=Xj;
        for j=1:length(Xnext)
            if Xnext(j)>lbub(2)
                Xnext(j)=lbub(2);
            elseif Xnext(j)<lbub(1)
                Xnext(j)=lbub(1);
            end
        end
    end
    Ynext=AF_rosenbrock(Xnext);
end