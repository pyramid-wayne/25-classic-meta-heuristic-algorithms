function y=fitness(xx)
    % 计算Rosenbrock目标函数
    [~,dim]=size(xx);
    x1=xx(:,1:dim-1);
    x2=xx(:,2:dim);
    if dim==2
        y=100.0*(x2-x1.^2).^2+(x1-1).^2;    % 2维
    else
        y=sum((100.0*(x2-x1.^2).^2+(x1-1).^2)')';
    end
end
