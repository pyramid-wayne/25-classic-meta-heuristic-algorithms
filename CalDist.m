function len=CalDist(dismat,solution)
    % º∆À„æ‡¿Î
    [soluNum,soluSize]=size(solution);
    len=zeros(1,soluNum);
    for i=1:soluNum
        R=solution(i,:);
        for j=1:soluSize-1
            len(i)=len(i)+dismat(R(j),R(j+1));
        end
        len(i)=len(i)+dismat(R(soluSize),R(1));
    end
end