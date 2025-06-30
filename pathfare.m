function objval=pathfare(fare,path)
    % 计算路径path的长度
    % path为1到n的排列，代表城市访问顺序
    [m,n]=size(path);
    objval=zeros(1,m);
    for i=1:m
        for j=2:n
            objval(i)=objval(i)+fare(path(i,j-1),path(i,j));
        end
        objval(i)=objval(i)+fare(path(i,n),path(i,1));
    end
end