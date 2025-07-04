function New_Sol=CombineRefset(x,y,num)
    % 合并子集
    if num==1
        d=randomD(x,y,4);
        New_Sol(1,:)=x-d(1,:);
        New_Sol(2,:)=x-d(2,:);
        New_Sol(3,:)=y+d(3,:);
        New_Sol(4,:)=x-d(4,:);
    elseif num==2
        d=randomD(x,y,3);
        New_Sol(1,:)=x+d(1,:);
        if d(3,:)<=0.5
            New_Sol(2,:)=x-d(2,:);
        else
            New_Sol(2,:)=y+d(2,:);
        end
    else
        d=randomD(x,y,3);
        New_Sol(1,:)=x-d(1,:);
        New_Sol(2,:)=x+d(2,:);
        New_Sol(3,:)=y+d(3,:);
    end
end