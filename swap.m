function newpath=swap(oldpath,number,position)
    % 完成邻域交换
    % 实现oldpath互换操作
    % number为互换的个数
    % position为对应newpath互换的位置
    m=length(oldpath);  % 城市个数
    newpath=zeros(number,m);  % 初始化newpath
    for i=1:number
        newpath(i,:)=oldpath;
        newpath(i,position(i,1))=oldpath(position(i,2));        % 互换位置
        newpath(i,position(i,2))=oldpath(position(i,1));
    end
    
end