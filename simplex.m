function [x,y,X,Y]=simplex(func,x,opts)
    % 单纯形法求解多维无约束非线性规划问题
    % 通过函数句柄FUNC从起始点X开始搜寻函数的局部最小值
    % 该算法不需要任何梯度信息，允许通过带有成员的结构Opts指定优化参数
    % OPTS的成员包括：
        % opts.Chi 控制扩展步骤的参数
        % opts.Delta 初始单纯形的参数控制大小
        % opts.Gamma 控制收缩步骤的参数
        % opts.Rho   控制反射步骤的参数
        % opts.Sigma 控制扩展步骤的参数
        % opts.MaxIter 最大迭代次数
        % opts.MaxFunEvals 函数求值的最大数量
        % opts.TolFun  基于每个步骤中函数值的相对变化的终止准则
        % opts.TolX 基于每个步骤中变量变化的终止准则
    % 输出参数：
        % x  返回的最小化解
        % y  返回的最小化值
        % X  优化工程中评估的所有X值
        % Y  返回函数的相应值
    % 初始化参数
    x=x(:);
    n=length(x);
    x=repmat(x',n+1,1);
    y=zeros(n+1,1);
    for i=1:n
        x(i,i)=x(i,i)+opts.Delta;
        y(i)=feval(func,x(i,:));
    end
    y(n+1)=feval(func,x(n+1,:));
    X=x;
    Y=y;
    count=n+1;
    % 迭代
    for i=2:opts.MaxIter
        [y,idx]=sort(y);        % order
        x=x(idx,:);
        centroid=mean(x(1:end-1,:));    % 计算中心点  reflect
        x_r=centroid+opts.Rho*(centroid-x(end,:));
        y_r=feval(func,x_r);
        count=count+1;
        X=[ X; x_r];
        Y=[ Y; y_r];
        if y_r>=y(1) && y_r<y(end-1)
            x(end,:)=x_r;
            y(end)=y_r;
        else
            if y_r<y(1)    % expand
                x_e=centroid+opts.Chi*(x_r-centroid);   % 计算扩展点
                y_e=feval(func,x_e);
                count=count+1;
                X=[ X; x_e];
                Y=[ Y; y_e];
                if y_e<y_r
                    x(end,:)=x_e;       % 扩展点更优----accept expansion point
                    y(end)=y_e;
                else
                    x(end,:)=x_r;       % 扩展点不优----accept reflection point
                    y(end)=y_r;
                end
            else    % contract
                shrink=0;       % contract
                if y(end-1)<=y_r && y_r<y(end)
                    x_c=centroid+opts.Gamma*(x_r-centroid);   % 计算收缩点 contract outside
                    y_c=feval(func,x_c);
                    count=count+1;
                    X=[ X; x_c];
                    Y=[ Y; y_c];
                    if y_c<y(end)
                        x(end,:)=x_c;       % 收缩点更优----accept contraction point
                        y(end)=y_c;
                    else
                        shrink=1;       % 收缩点不优----shrink
                    end
                else
                    x_c=centroid+opts.Gamma*(centroid-x(end,:));   % 计算收缩点 contract inside
                    y_c=feval(func,x_c);
                    count=count+1;
                    X=[ X; x_c];
                    Y=[ Y; y_c];
                    if y_c<y(end)
                        x(end,:)=x_c;       % 收缩点更优----accept contraction point
                        y(end)=y_c;
                    else
                        shrink=1;       % 收缩点不优----shrink
                    end
                end
                if shrink==1
                    for j=2:n+1
                        x(j,:)=x(1,:)+opts.Sigma*(x(j,:)-x(1,:));
                        y(j)=feval(func,x(j,:));
                        count=count+1;
                        X=[ X; x(j,:)];
                        Y=[ Y; y(j)];
                    end
                end
            end
        end

        if max(abs(min(x)-max(x)))<opts.TolX
            break;
        end

        if abs(max(y)-min(y))/max(abs(y))<opts.TolFun
            break;
        end
    end
    [y,idx]=min(y);     % update model structure
    x=x(idx,:);
end