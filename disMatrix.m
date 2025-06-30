function D=disMatrix(n,Coord)
    % 计算距离矩阵
    % 输入：n-节点数，Coord-坐标矩阵
    D=zeros(n,n);
    for i=1:n
        for j=i:n
            D(i,j)=((Coord(i,1)-Coord(j,1))^2+(Coord(i,2)-Coord(j,2))^2)^0.5;
            D(j,i)=D(i,j);
        end
    end
end
