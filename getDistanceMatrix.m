function  distanceMatrix = getDistanceMatrix( cityNum, city )
    % ������������֮��ľ��������
    distanceMatrix = zeros(cityNum, cityNum);
    for i = 1:cityNum
        for j = 1:cityNum
            distanceMatrix(i,j) = sqrt((city(i,1)-city(j,1))^2 + (city(i,2)-city(j,2))^2);
        end
    end
end