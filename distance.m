function fare=distance(coord)
    % 计算距离矩阵
    [n,m]=size(coord);  % m为城市的个数
    fare=zeros(m);
    for x=1:m       % 外层行
        for y=x:m   % 内层列
            fare(x,y)=sum((coord(:,x)-coord(:,y)).^2)^0.5;
            fare(y,x)=fare(x,y);    % 距离对称矩阵
        end
    end
end
