function rank=ranking(Xdata,city,probSize)
    % ���վ���Ե�ǰ�ڵ�����
    rank=zeros(probSize,2);
    rank(:,1)=Xdata(:,city);
    rank(:,2)=1:probSize;
end