function neighbor=neighborhoodOne(sol)
    % ����ṹ1��swap����
    len=length(sol);
    count=1:len-1;
    neighborNum=sum(count);     % �������
    neighbor=zeros(neighborNum,len);
    k=0;
    for i=1:len
        for j=i+1:len
            k=k+1;
            s=sol;
            x=s(:,j);
            s(:,j)=s(:,i);
            s(:,i)=x;           % �����µ������
            neighbor(k,:)=s;    % �洢�µ������
        end
    end
end