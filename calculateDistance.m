function distance=calculateDistance(solution,distanceMatrix)
    % º∆À„Ω‚solutionµƒæ‡¿Î
    DistanceV=0;
    n=size(solution,2);
    for i=1:n-1
        DistanceV=DistanceV+distanceMatrix(solution(i),solution(i+1));
    end
    DistanceV=DistanceV+distanceMatrix(solution(n),solution(1));
    distance=DistanceV;
end