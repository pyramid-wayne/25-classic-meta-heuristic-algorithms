function [RCL,sequence] = RCL(Xdata,alfa,probSize)
    % 贪心随机搜索，构造RCL
    start=randperm(probSize,1);     % 随机选择一个初始点
    sequence=start;               % 记录搜索路径
    while length(sequence)<probSize     % 当搜索路径长度小于种群规模时
        rand1=rand();           % 产生一个随机数
        if rand1>alfa           % 贪心选择
            if sequence(end)==1     % 排除节点为1的值
                city=1;
            else
                city=sequence(end)-1;  % 选择前一个节点
            end
            rank=ranking(Xdata,city,probSize);    % 返回当前节点的值
            [~,I]=sort(rank(:,1));      % 排序
            rank=rank(I,:);             % 排序后的矩阵
            count=0;                    % 选择最小值
            next_city=rank(1,2);        % 初始化下一个节点 
            while ismember(next_city,sequence)  % 若已在当前解序列予以排除
                count=count+1;
                next_city=rank(count,2);        % 选择下一个节点
            end
            sequence=[sequence,next_city];      % 将下一个节点加入当前解序列
        else    % 随机旋转
            next_city=randperm(probSize,1);       % 随机选择一个节点
            while ismember(next_city,sequence)    % 若已在当前解序列予以排除
                next_city=randperm(probSize,1);   % 随机选择一个节点
            end
            sequence=[sequence,next_city];        % 将下一个节点加入当前解序列
        end
    end
    sequence=[sequence,sequence(1)];            % 回到原点
    RCL=distance_calc(Xdata,sequence);          % 计算路径长度        
end