function neighbor=neighborhoodThree(sol)
    % ����ṹ1��two_h_opt_swap����
    len=length(sol);
    count=1:len-1;           % ʣ�ಽ��
    neighborNum=sum(count);     % �������
    neighbor=zeros(neighborNum,len);
    k=0;
    for i=1:len
        for j=i+1:len
            k=k+1;
            s=sol;
            s=[s(i),s(j),s(1:i-1),s(i+1:j-1),s(j+1:end)];    % ��������λ��---�����µ������
            neighbor(k,:)=s;    % �洢�µ������
        end
    end
end