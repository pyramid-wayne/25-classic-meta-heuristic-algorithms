function L=RandSample(P,q)
    % 从子群中随机选择q个解样本
    if ~exist('replacement','var')
        replacement = false;
    end
    L=zeros(q,1);
    for i=1:q
        L(i)=randsample(numel(P),1,true,P); % 从P中随机选择一个元素
        if ~replacement
            P(L(i))=0;
        end
    end
end