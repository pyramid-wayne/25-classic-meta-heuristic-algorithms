function y=fitness(xx)
    % ����RosenbrockĿ�꺯��
    [~,dim]=size(xx);
    x1=xx(:,1:dim-1);
    x2=xx(:,2:dim);
    if dim==2
        y=100.0*(x2-x1.^2).^2+(x1-1).^2;    % 2ά
    else
        y=sum((100.0*(x2-x1.^2).^2+(x1-1).^2)')';
    end
end
