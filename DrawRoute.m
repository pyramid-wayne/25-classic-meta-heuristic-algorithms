function DrawRoute(C,R)
    % ��������·��ͼ
    N=length(R);
    scatter(C(:,1),C(:,2));     % ������ɢ����
    set(gcf,'color','none');
    alpha(0.1)
    hold on;
    plot([C(R(1),1),C(R(N),1)],[C(R(1),2),C(R(N),2)],'r');  % ����·��
    hold on;
    for i=2:N
        plot([C(R(i-1),1),C(R(i),1)],[C(R(i-1),2),C(R(i),2)],'k');
        hold on;            % ����������������
    end
    title('�Ż����·��ͼ');