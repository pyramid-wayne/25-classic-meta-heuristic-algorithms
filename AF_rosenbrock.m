function y=AF_rosenbrock(xx)
    % 计算rosenbrock 目标函数值
    [~,dim]=size(xx);
    x1=xx(:,dim-1);
    x2=xx(:,2:dim);
    if dim==2
        y=100*(x2-x1.^2).^2+(1-x1).^2; % 两个变量
    else
        y=sum((100*(x2-x1.^2).^2+(1-x1).^2)')'; % 多个变量
    end
end