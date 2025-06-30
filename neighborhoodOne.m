function neighbor=neighborhoodOne(sol)
    % 邻域结构1：swap算子
    len=length(sol);
    count=1:len-1;
    neighborNum=sum(count);     % 邻域个数
    neighbor=zeros(neighborNum,len);
    k=0;
    for i=1:len
        for j=i+1:len
            k=k+1;
            s=sol;
            x=s(:,j);
            s(:,j)=s(:,i);
            s(:,i)=x;           % 产生新的邻域解
            neighbor(k,:)=s;    % 存储新的邻域解
        end
    end
end