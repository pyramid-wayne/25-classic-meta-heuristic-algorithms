function y=AF_rosenbrock(xx)
    % ����rosenbrock Ŀ�꺯��ֵ
    [~,dim]=size(xx);
    x1=xx(:,dim-1);
    x2=xx(:,2:dim);
    if dim==2
        y=100*(x2-x1.^2).^2+(1-x1).^2; % ��������
    else
        y=sum((100*(x2-x1.^2).^2+(1-x1).^2)')'; % �������
    end
end