function NB=Neiborhood(cityNum,neiborNum)
    % 产生一组2-opt 邻域交换位置
    ik=1;
    NB=zeros(neiborNum,2);  % 保存邻域
    while ik<=neiborNum     % 遍历所有邻域，直到产生neiborNum个邻域为止
        M=ceil(cityNum*rand(1,2));  % 随机产生两个不重复位置
        if M(1)~=M(2)
            NB(ik,1)=min(M);
            NB(ik,2)=max(M);
            if ik==1
                isdel=0;        % 第一个邻域交换不需要检测
            else
                for jk=1:ik-1   % 其余邻域交换需要检测是否重复
                    if NB(ik,1)==NB(jk,1) && NB(ik,2)==NB(jk,2)
                        isdel=1;    % 如果重复，则删除,重复标记
                        break;
                    else
                        isdel=0;    % 如果不重复，则继续
                    end
                end
            end
            if ~isdel
                ik=ik+1;       % 如果不重复，则继续
            else
                continue;        % 如果重复，则重新随机产生
            end
        else
            continue;        % 如果随机产生位置相同，则重新随机产生
        end
    end
end