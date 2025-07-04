function [New_Refset1,New_Solt]=Update_RefSet1(New_Sol,refset11)
    % 更新参考集Refset
    b1=size(refset11,1);
    [New_Solt,~]=sortrows(New_Sol,size(New_Sol,2));     % 依据目标值大小排序
    temp1=[];
    temp1=[temp1;New_Solt(1:b1,:);refset11];
    [refset1,~]=sortrows(temp1,size(temp1,2));     % 依据目标值大小排序
    New_Refset1=refset1(1:b1,:);            % 取前b1个高质量解
    
end