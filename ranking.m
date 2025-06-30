function rank=ranking(Xdata,city,probSize)
    % 按照距离对当前节点排序
    rank=zeros(probSize,2);
    rank(:,1)=Xdata(:,city);
    rank(:,2)=1:probSize;
end