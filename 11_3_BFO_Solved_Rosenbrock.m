% 细菌觅食算法求解Rosenbrock函数，BFO: Bacterial Foraging Optimization
% 目标解；[1,1,1,1,1];维度：5，5个变量的Rosenbrock函数，最小值的解
clc;clear;close;
p=5;            % 变量个数
bactNum=300;    % 初始细菌个数
Nc=50;          % 趋化性次数，chemotactic
Ns=40;          % 游动次数
Nre=10;         % 复制次数；reproduction
Ned=8;          % 迁徙次数：elimation and 
Sr=bactNum/2;   % 一袋细菌被复制个数
Ped=0.25;       % 细菌消亡率
d_attract=0.0005; % 吸引度和排斥力参数
ommiga_attract=0.0005;
h_repellant=0.0005;
cMax=0.11;     % 最大游动步长
Delta=zeros(p,bactNum); % 邻域变换补偿
solution=zeros(p,bactNum,Nc,Nre,Ned);       % 目标解
% 目标函数bactNum只细菌做Nc次趋化操作，Nre次复制，Ned次消亡
J=zeros(bactNum,Nc,Nre,Ned);        % 目标函数值
% 初始化细菌位置
for m=1:p
    for i=1:bactNum
        solution(m,i,1,1,1)=6*rand-3;   % -3,3区间
    end
end
globalBestValue=Inf;        % 全局最优值
for dis=1:Ned       % 消亡，最外层循环
    for K=1:Nre      % 复制，中间循环
        for j=1:Nc   % 趋化，最内层循环
            for i=1:bactNum % 遍历每个细菌
                J(i,j,K,dis)=Rosenbrock(solution(:,i,j,K,dis)');     % 计算目标函数值
                currentBestValue=J(i,j,K,dis);
                currentBestSolution=solution(:,i,j,K,dis);
                sol=0;  % 计算调整值Jcc
                for weishu=1:p
                    for bac=1:bactNum
                        if bac~=i
                            sol=sol+(solution(weishu,i,j,K,dis)-solution(weishu,bac,j,K,dis)).^2;
                        end
                    end
                end
                Jcc=sum(-d_attract*exp(-ommiga_attract*sol))+sum(h_repellant*exp(-ommiga_attract*sol));
                J(i,j,K,dis)=J(i,j,K,dis)+Jcc;
                c(:,K)=cMax-0.1*(ones(bactNum,1)-(Nc-j)/Nc); % 可变游动步长 0.11~0.01
                Delta(:,i)=(2*round(rand(p,1))-1).*rand(p,1); % 邻域变换补偿
                solution(:,i,j+1,K,dis)=solution(:,i,j,K,dis)+c(i,K)*Delta(:,i)/sqrt(Delta(:,i)'*Delta(:,i)); % 旋转
                J(i,j+1,K,dis)=Rosenbrock(solution(:,i,j+1,K,dis)'); % 计算目标函数值

                m=0;            % 初始化游动次数
                while m<Ns      % 细菌游动Ns次
                    m=m+1;
                    if J(i,j+1,K,dis)<currentBestValue
                        currentBestValue=J(i,j+1,K,dis);    % 更新当前细菌最优值
                        currentBestSolution=solution(:,i,j+1,K,dis); % 更新当前细菌最优解
                        solution(:,i,j+1,K,dis)=solution(:,i,j+1,K,dis)+c(i,K)*Delta(:,i)/sqrt(Delta(:,i)'*Delta(:,i));     % 游动一次
                        J(i,j+1,K,dis)=Rosenbrock(solution(:,i,j+1,K,dis)'); % 计算目标函数值
                    else
                        m=Ns;               % 若沿当前方向搜索没有改善，则停止游动
                    end
                end
                % 更新当前细菌最优值和最优解
                if currentBestValue<globalBestValue     % 与当前最佳目标值比较
                    globalBestValue=currentBestValue;   % 保存最佳目标值
                    globalBestSolution=currentBestSolution; % 保存最佳解
                end
            end % 转到下一只细菌

            % 动态画图，消耗时间
            x=solution(1,:,j,K,dis);
            y=solution(2,:,j,K,dis);
            clf;
            plot(x,y,'h')
            axis([-5 5 -5 5]);
            pause(0.05);
        end % 转到下一轮趋化

        % 复制细菌，复制新的种群
        temp=[];
        solution1=[];
        for comb=1:Nc+1
            temp=[temp,J(:,comb,K,dis)];        % 目标值和目标解的整理
            solution1=[solution1, solution(:,:,comb,K,dis)];
        end
        [~,ind]=sort(temp);        % 排序 目标值按大小
        solution2=solution1(:,ind); % 排序后的目标解，目标解按索引调整顺序
        solution(:,:,1,K+1,dis)=solution2(:,1:bactNum); % 按照索引保留前bactNum个最优解
        for i=1:Sr  % 复制一半优秀的个体到另一半
            solution(:,i+Sr,j,K+1,dis)=solution(:,i,1,K+1,dis);
            c(i+Sr,K+1)=c(i,K);
        end
    end % 转到下一轮复制
    
    for m=1:p
        for i=1:bactNum
            if Ped>rand
                solution(m,i,1,1,1)=6*rand-3;   % -3,3区间
            end
        end
    end
end     % 转到下一轮消亡 Ned
% 输出结果
globalBestValue
globalBestSolution









