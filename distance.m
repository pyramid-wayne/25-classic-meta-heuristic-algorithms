function fare=distance(coord)
    % ����������
    [n,m]=size(coord);  % mΪ���еĸ���
    fare=zeros(m);
    for x=1:m       % �����
        for y=x:m   % �ڲ���
            fare(x,y)=sum((coord(:,x)-coord(:,y)).^2)^0.5;
            fare(y,x)=fare(x,y);    % ����Գƾ���
        end
    end
end
