function DrawRoute(C,R)
    % 画出最终路线图
    N=length(R);
    scatter(C(:,1),C(:,2));     % 画出离散坐标
    set(gcf,'color','none');
    alpha(0.1)
    hold on;
    plot([C(R(1),1),C(R(N),1)],[C(R(1),2),C(R(N),2)],'r');  % 画出路线
    hold on;
    for i=2:N
        plot([C(R(i-1),1),C(R(i),1)],[C(R(i-1),2),C(R(i),2)],'k');
        hold on;            % 逐条画出其他连线
    end
    title('优化后的路线图');