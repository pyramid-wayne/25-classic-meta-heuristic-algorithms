function d=randomD(x,y,num)
    % �������������
    d=[];
    for i=1:num
        s=rand*(x-y)/2;
        d=[d;s];
    end
end