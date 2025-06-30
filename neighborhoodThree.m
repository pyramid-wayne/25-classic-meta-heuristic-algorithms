function neighbor=neighborhoodThree(sol)
    % 邻域结构1：two_h_opt_swap算子
    len=length(sol);
    count=1:len-1;           % 剩余步长
    neighborNum=sum(count);     % 邻域个数
    neighbor=zeros(neighborNum,len);
    k=0;
    for i=1:len
        for j=i+1:len
            k=k+1;
            s=sol;
            s=[s(i),s(j),s(1:i-1),s(i+1:j-1),s(j+1:end)];    % 交换两个位置---产生新的邻域解
            neighbor(k,:)=s;    % 存储新的邻域解
        end
    end
end