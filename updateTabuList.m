function tabuList=updateTabuList(tabuList,x,y,cityNum,tabuListLength)
    % ���½��ɱ�������Ԫ�أ����������һ
    for m=1:cityNum
        for n=1:cityNum
            if tabuList(m,n)==0
                tabuList(m,n)=tabuList(m,n)-1;
            end
        end
    end
    tabuList(x,y)=tabuListLength;
end