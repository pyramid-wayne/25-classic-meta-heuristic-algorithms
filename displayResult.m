function  displayResult(currentTime,bestSolution,bestFitnessValue,cityNum,city)
    % ��ʾ���·���仯
    for i=1:cityNum-1
        plot([city(bestSolution(i),1),city(bestSolution(i+1),1)],[city(bestSolution(i),2),city(bestSolution(i+1),2)],'bo-');
        hold on;
    end
    plot([city(bestSolution(cityNum),1),city(bestSolution(1),1)],[city(bestSolution(cityNum),2),city(bestSolution(1),2)],'bo-');
    title(['counter:',int2str(currentTime),'The Min Distance'  ,int2str(bestFitnessValue)]);
    hold off;
end