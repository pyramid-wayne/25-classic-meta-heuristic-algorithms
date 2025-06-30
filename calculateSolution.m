function solutionValue=calculateSolution(empBeeNum,D,cityNum,Employed)
    % 计算目标解
    solutionValue=zeros(empBeeNum,1);
    for i=1:empBeeNum
        R=Employed(i,:);
        for j=1:cityNum-1
            solutionValue(i)=solutionValue(i)+D(R(j),R(j+1));
        end
        solutionValue(i)=solutionValue(i)+D(R(cityNum),R(1));   % 回到起点
    end
end