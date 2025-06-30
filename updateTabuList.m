function tabuList=updateTabuList(tabuList,x,y,cityNum,tabuListLength)
    % 更新禁忌表，加入新元素，其余遍历减一
    for m=1:cityNum
        for n=1:cityNum
            if tabuList(m,n)==0
                tabuList(m,n)=tabuList(m,n)-1;
            end
        end
    end
    tabuList(x,y)=tabuListLength;
end