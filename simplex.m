function [x,y,X,Y]=simplex(func,x,opts)
    % �����η�����ά��Լ�������Թ滮����
    % ͨ���������FUNC����ʼ��X��ʼ��Ѱ�����ľֲ���Сֵ
    % ���㷨����Ҫ�κ��ݶ���Ϣ������ͨ�����г�Ա�ĽṹOptsָ���Ż�����
    % OPTS�ĳ�Ա������
        % opts.Chi ������չ����Ĳ���
        % opts.Delta ��ʼ�����εĲ������ƴ�С
        % opts.Gamma ������������Ĳ���
        % opts.Rho   ���Ʒ��䲽��Ĳ���
        % opts.Sigma ������չ����Ĳ���
        % opts.MaxIter ����������
        % opts.MaxFunEvals ������ֵ���������
        % opts.TolFun  ����ÿ�������к���ֵ����Ա仯����ֹ׼��
        % opts.TolX ����ÿ�������б����仯����ֹ׼��
    % ���������
        % x  ���ص���С����
        % y  ���ص���С��ֵ
        % X  �Ż�����������������Xֵ
        % Y  ���غ�������Ӧֵ
    % ��ʼ������
    x=x(:);
    n=length(x);
    x=repmat(x',n+1,1);
    y=zeros(n+1,1);
    for i=1:n
        x(i,i)=x(i,i)+opts.Delta;
        y(i)=feval(func,x(i,:));
    end
    y(n+1)=feval(func,x(n+1,:));
    X=x;
    Y=y;
    count=n+1;
    % ����
    for i=2:opts.MaxIter
        [y,idx]=sort(y);        % order
        x=x(idx,:);
        centroid=mean(x(1:end-1,:));    % �������ĵ�  reflect
        x_r=centroid+opts.Rho*(centroid-x(end,:));
        y_r=feval(func,x_r);
        count=count+1;
        X=[ X; x_r];
        Y=[ Y; y_r];
        if y_r>=y(1) && y_r<y(end-1)
            x(end,:)=x_r;
            y(end)=y_r;
        else
            if y_r<y(1)    % expand
                x_e=centroid+opts.Chi*(x_r-centroid);   % ������չ��
                y_e=feval(func,x_e);
                count=count+1;
                X=[ X; x_e];
                Y=[ Y; y_e];
                if y_e<y_r
                    x(end,:)=x_e;       % ��չ�����----accept expansion point
                    y(end)=y_e;
                else
                    x(end,:)=x_r;       % ��չ�㲻��----accept reflection point
                    y(end)=y_r;
                end
            else    % contract
                shrink=0;       % contract
                if y(end-1)<=y_r && y_r<y(end)
                    x_c=centroid+opts.Gamma*(x_r-centroid);   % ���������� contract outside
                    y_c=feval(func,x_c);
                    count=count+1;
                    X=[ X; x_c];
                    Y=[ Y; y_c];
                    if y_c<y(end)
                        x(end,:)=x_c;       % ���������----accept contraction point
                        y(end)=y_c;
                    else
                        shrink=1;       % �����㲻��----shrink
                    end
                else
                    x_c=centroid+opts.Gamma*(centroid-x(end,:));   % ���������� contract inside
                    y_c=feval(func,x_c);
                    count=count+1;
                    X=[ X; x_c];
                    Y=[ Y; y_c];
                    if y_c<y(end)
                        x(end,:)=x_c;       % ���������----accept contraction point
                        y(end)=y_c;
                    else
                        shrink=1;       % �����㲻��----shrink
                    end
                end
                if shrink==1
                    for j=2:n+1
                        x(j,:)=x(1,:)+opts.Sigma*(x(j,:)-x(1,:));
                        y(j)=feval(func,x(j,:));
                        count=count+1;
                        X=[ X; x(j,:)];
                        Y=[ Y; y(j)];
                    end
                end
            end
        end

        if max(abs(min(x)-max(x)))<opts.TolX
            break;
        end

        if abs(max(y)-min(y))/max(abs(y))<opts.TolFun
            break;
        end
    end
    [y,idx]=min(y);     % update model structure
    x=x(idx,:);
end