function D =AF_dist(Xi,X)
    % 计算范数距离
    [row,~] = size(X);
    D=zeros(1,row);
    for i=1:row
        D(i) = norm(Xi-X(i,:));     % d等效于norm(A,2)
    end
end