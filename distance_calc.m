function distanceV=distance_calc(Xdata,city_tour)
    % ����Ŀ�꺯��ֵ
    distanceV=0;
    n=size(city_tour,2);
    for i=1:n-1
        distanceV=distanceV+Xdata(city_tour(i),city_tour(i+1));
    end
    distanceV=distanceV+Xdata(city_tour(n),city_tour(1)); % �ص�ԭ��
end