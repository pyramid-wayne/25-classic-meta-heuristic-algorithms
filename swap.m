function newpath=swap(oldpath,number,position)
    % ������򽻻�
    % ʵ��oldpath��������
    % numberΪ�����ĸ���
    % positionΪ��Ӧnewpath������λ��
    m=length(oldpath);  % ���и���
    newpath=zeros(number,m);  % ��ʼ��newpath
    for i=1:number
        newpath(i,:)=oldpath;
        newpath(i,position(i,1))=oldpath(position(i,2));        % ����λ��
        newpath(i,position(i,2))=oldpath(position(i,1));
    end
    
end