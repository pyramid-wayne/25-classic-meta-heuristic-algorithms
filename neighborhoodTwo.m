function neighbor=neighborhoodTwo(sol)
    % ����ṹ1��two_opt_swap����
    len=length(sol);
    step=3;
    count=1:len-step;           % ʣ�ಽ��
    neighborNum=sum(count);     % �������
    neighbor=zeros(neighborNum,len);
    k=0;
    for i=1:len
        for j=i+3:len
            k=k+1;
            s=sol;
            s1=s(i:j);
            s1=fliplr(s1);
            s=[s(1:i-1),s1,s(j+1:end)];
            neighbor(k,:)=s;    % �洢�µ������
        end
    end
end