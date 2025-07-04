function d=randomD(x,y,num)
    % 随机数矩阵生成
    d=[];
    for i=1:num
        s=rand*(x-y)/2;
        d=[d;s];
    end
end