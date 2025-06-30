function neighbor=neighborhoodTwo(sol)
    % 邻域结构1：two_opt_swap算子
    len=length(sol);
    step=3;
    count=1:len-step;           % 剩余步长
    neighborNum=sum(count);     % 邻域个数
    neighbor=zeros(neighborNum,len);
    k=0;
    for i=1:len
        for j=i+3:len
            k=k+1;
            s=sol;
            s1=s(i:j);
            s1=fliplr(s1);
            s=[s(1:i-1),s1,s(j+1:end)];
            neighbor(k,:)=s;    % 存储新的邻域解
        end
    end
end