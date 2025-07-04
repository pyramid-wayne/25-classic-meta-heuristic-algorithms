function [Xnext,Ynext]=AF_swarm(X,i,visual,step,delta,try_num,lbub,lastY)
    % ��Ⱥ��ʳ����
    Xi=X(i,:);              % ȡ����ǰ��
    D=AF_dist(Xi,X);        % ����norm(Xi-X)����
    index=find(D>0& D<visual);      % ��ȥ������Ӿ�֮����˹�Ⱥ
    nf=length(index);               % ���������������˹���Ⱥ����
    if nf>0
        for j=1:size(X,2)
            temp=X(:,j);
            Xc(j)=mean(temp(index));    % ���������������˹�Ⱥ����
        end
        Yc=AF_rosenbrock(Xc);           % ���������������˹����Ŀ��ֵ
        Yi=lastY(i);                    % ȡ����ǰ���Ŀ��ֵ
        if Yc/nf<delta*Yi               % �ж��Ƿ�������ʳ����
            Xnext=Xi+rand*step*(Xc-Xi)/norm(Xc-Xi);     % ��ǰλ��������λ���ƶ�
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
            [Xnext,Ynext]=AF_prey(Xi,i,visual,step,try_num,lbub,lastY);     % ����
        end
    else
        [Xnext,Ynext]=AF_prey(Xi,i,visual,step,try_num,lbub,lastY);         % ����
    end
end