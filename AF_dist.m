function D =AF_dist(Xi,X)
    % ���㷶������
    [row,~] = size(X);
    D=zeros(1,row);
    for i=1:row
        D(i) = norm(Xi-X(i,:));     % d��Ч��norm(A,2)
    end
end