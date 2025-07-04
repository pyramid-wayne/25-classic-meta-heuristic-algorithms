function [Xnext,Ynext]=AF_prey(Xi,ii,visual,step,try_num,lbub,lastY)
    % ������ʳ����
    Xnext=[];
    Yi=lastY(ii);
    for i=1:try_num     % �����Զ��
        Xj=Xi+((2*rand(length(Xi),1)-1)*visual)';
        Yj=AF_rosenbrock(Xj);
        if Yj<Yi        % �ж��ƶ�Ч��
            Xnext=Xi+rand*step*(Xj-Xi)/norm(Xj-Xi); % �����ƶ�һ��
            for j=1:length(Xnext)
                if Xnext(j)>lbub(2)
                    Xnext(j)=lbub(2);
                elseif Xnext(j)<lbub(1)
                    Xnext(j)=lbub(1);
                end
            end
            Xi=Xnext;
            break;
        end
    end
    % �����Ϊ
    if isempty(Xnext)       % ���Զ��֮��δ�����仯���������ִ��һ������仯
        Xj=Xi+((2*rand(length(Xi),1)-1)*visual)';
        Xnext=Xj;
        for j=1:length(Xnext)
            if Xnext(j)>lbub(2)
                Xnext(j)=lbub(2);
            elseif Xnext(j)<lbub(1)
                Xnext(j)=lbub(1);
            end
        end
    end
    Ynext=AF_rosenbrock(Xnext);
end