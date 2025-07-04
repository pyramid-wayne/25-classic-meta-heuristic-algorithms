function [Xnext,Ynext] = AF_follow(X,i,visual,step,delta,try_num,lbub,lastY)
    % ׷β��ʳ����
    Xi=X(i,:);      % ȡ����ǰ��
    D=AF_dist(Xi,X);    % ����norm(Xi-X)����
    index=find(D>0&D<visual);    % �ҳ�������[0,visual]��Χ�ڵĸ���
    nf=length(index);    % ���������[0,visual]��Χ�ڵĸ������
    if nf>0
        XX=X(index,:);
        YY=lastY(index);
        [Ymin,min_index]=min(YY);       % �����������˹������Сλ��
        Xmin=XX(min_index,:);
        Yi=lastY(i);
        if Ymin/nf<delta*Yi     % �����ʽ��������תȥִ�г�����ʳ��Ϊ
            Xnext=Xi+rand*step*(Xmin-Xi)/norm(Xmin-Xi);    % ��ǰλ�þ���λ���ƶ�һ��
            for i=1:length(Xnext)
                if Xnext(i)>lbub(2)
                    Xnext(i)=lbub(2);
                end
                if Xnext(i)<lbub(1)
                    Xnext(i)=lbub(1);
                end
            end
            Ynext=AF_rosenbrock(Xnext);
        else
            [Xnext,Ynext]=AF_prey(X(i,:),i,visual,step,try_num,lbub,lastY);    % ִ�г�����ʳ��Ϊ
        end
    else
        [Xnext,Ynext]=AF_prey(X(i,:),i,visual,step,try_num,lbub,lastY);    % ִ�г�����ʳ��Ϊ
    end
end