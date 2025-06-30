% 进化规划EP求解二元函数:y=21.5+x1*sin(4*pi*x1)+x2*sin(20*pi*x2)最大值，x1∈[-0.3,12.1],x2∈[4.1,5.8]
% 种群规模：40；迭代次数：200；
% x1=10.4837,x2=5.2850,maxy=38.8485
clc;clear;close all;
% 初始化参数
miu=100;                    % 种群规模
MaxIter=1000;               % 最大迭代次数
sigma_x1=ones(1,miu);       % 每个个体x1分量标准差初始值
sigma_x2=ones(1,miu);       % 每个个体x2分量标准差初始值
compute_fit=@( x1,x2 ) 21.5+x1.*sin(4*pi.*x1)+x2.*sin(20*pi.*x2); % 适应度函数
rand('state',sum(100*clock)); % 重置随机数种子生成器
pop_x1=zeros(1,miu);          % 存放每代个体x1分量的值
pop_x2=zeros(1,miu);          % 存放每代个体x2分量的值
for j=1:5           % 平均变异
    for i=1:20
        pop_x1((j-1)*20+i)=-0.3+(i-1)*0.62+rand*0.62;   % x1分为20等分
        pop_x2((j-1)*20+i)=4.1+(j-1)*0.34+rand*0.34;   % x2分为20等分
    end
end
fit_pop=compute_fit(pop_x1,pop_x2);  % 计算初始种群适应度
fit_output=sort(fit_pop,'descend');           % 对适应度排序---降序
for g=1:MaxIter
    pop_mutate_x1=pop_x1+sqrt(sigma_x1).*normrnd(0,1,[1,miu]); % 变异个体x1分量
    for k=1:miu
        while pop_mutate_x1(k)<-0.3 || pop_mutate_x1(k)>12.1 % 变异个体x1分量越界处理
            pop_mutate_x1(k)=pop_x1(k)+sqrt(sigma_x1(k)).*normrnd(0,1);
        end
    end
    pop_mutate_x2=pop_x2+sqrt(sigma_x2).*normrnd(0,1,[1,miu]); % 变异个体x2分量
    for k=1:miu
        while pop_mutate_x2(k)<4.1 || pop_mutate_x2(k)>5.8 % 变异个体x2分量越界处理
            pop_mutate_x2(k)=pop_x2(k)+sqrt(sigma_x2(k)).*normrnd(0,1);
        end
    end
    fit_mutate=compute_fit(pop_mutate_x1,pop_mutate_x2);    % 计算变异个体适应度
    sigma_x1=sigma_x1+sqrt(sigma_x1).*normrnd(0,1,[1,miu]); % 变异个体x1分量标准差更新
    sigma_x1=abs(sigma_x1);                                 % 确保标准差为正
    sigma_x2=sigma_x2+sqrt(sigma_x2).*normrnd(0,1,[1,miu]); % 变异个体x2分量标准差更新
    sigma_x2=abs(sigma_x2);                                 % 确保标准差为正
    % 采用q竞争法选择个体组成新种群
    pop_temp_x1=cat(2,pop_x1,pop_mutate_x1); % 合并父代和变异个体
    pop_temp_x2=cat(2,pop_x2,pop_mutate_x2); % 合并父代和变异个体
    fit_temp=compute_fit(pop_temp_x1,pop_temp_x2); % 父代和变异适应度
    score=zeros(1,miu*2); % 计算竞争得分
    for list=1:miu*2
        % 从1到200中随机选择90个不重复的数作为q竞争法选择的裁判的位置
        position=randperm(2*miu,0.9*miu);
        judge_x1=pop_temp_x1(position); % 裁判个体x1分量
        judge_x2=pop_temp_x2(position); % 裁判个体x2分量
        fit_judge=compute_fit(judge_x1,judge_x2); % 裁判适应度
        for m=1:0.9*miu
            if fit_temp(list)>fit_judge(m)
                score(list)=score(list)+1;
            end
        end
    end
    [score,location]=sort(score,'descend'); % 对得分排序---降序
    pop_x1=pop_temp_x1(location(1:miu)); % 挑选得分最好的新种群x1分量
    pop_x2=pop_temp_x2(location(1:miu)); % 挑选得分最好的新种群x2分量
    fit_pop=compute_fit(pop_x1,pop_x2);    % 计算新种群适应度
    fit_output=sort(fit_pop,'descend');    % 新种群对适应度排序---降序
    fprintf('第%d次迭代，最大适应度：%f\n',g,fit_output(1)); % 输出每次迭代最大适应度
end
