function  distanceMatrix = getDistanceMatrix( cityNum, city )
    % 计算两个城市之间的距离矩阵函数
    distanceMatrix = zeros(cityNum, cityNum);
    for i = 1:cityNum
        for j = 1:cityNum
            distanceMatrix(i,j) = sqrt((city(i,1)-city(j,1))^2 + (city(i,2)-city(j,2))^2);
        end
    end
end