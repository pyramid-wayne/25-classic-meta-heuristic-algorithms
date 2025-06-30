% 符号回归：目标函数 y = x^2 + x + 1
% Chatgpt示例
clear; clc;

% Step 1: 训练数据
x = linspace(-5,5,50)';
y = x.^2 + x + 1;

% Step 2: 设置 GP 参数
pop_size = 30;
max_gen = 20;
max_depth = 4;

% 函数集 & 终止符
functions = {@plus, @minus, @times};  % 基础三种操作
terminals = {@() x, @() rand()*10};   % 变量 & 随机常数

% 初始化种群
population = cell(pop_size,1);
for i = 1:pop_size
    population{i} = random_tree(functions, terminals, max_depth);
end

% Step 3: GP主循环
for gen = 1:max_gen
    fitness = zeros(pop_size,1);
    for i = 1:pop_size
        try
            y_pred = evaluate_tree(population{i});
            fitness(i) = mean((y_pred - y).^2);
        catch
            fitness(i) = inf;
        end
    end

    % 输出当前最优
    [best_fit, best_idx] = min(fitness);
    fprintf("Gen %d: Best MSE = %.4f\n", gen, best_fit);

    % Step 4: 选择+交叉+变异
    new_pop = cell(pop_size,1);
    for i = 1:pop_size
        % 锦标赛选择
        p1 = tournament_selection(population, fitness);
        p2 = tournament_selection(population, fitness);
        child = crossover(p1, p2);
        child = mutate(child, functions, terminals, max_depth);
        new_pop{i} = child;
    end
    population = new_pop;
end

% 显示最优表达式
disp('最优表达式树结构：')
pretty_print_tree(population{best_idx});

% 预测与绘图
y_best = evaluate_tree(population{best_idx});
plot(x, y, 'k', x, y_best, 'r--');
legend('真实', 'GP预测');
title('遗传编程拟合结果');

%% --- 子函数区域 ---

function node = random_tree(funcs, terms, depth)
    if depth == 0 || rand < 0.3
        node = terms{randi(length(terms))};
    else
        f = funcs{randi(length(funcs))};
        node = {f, random_tree(funcs, terms, depth-1), random_tree(funcs, terms, depth-1)};
    end
end

function y = evaluate_tree(tree)
    if isa(tree, 'function_handle')
        y = tree();
    elseif iscell(tree)
        f = tree{1};
        a = evaluate_tree(tree{2});
        b = evaluate_tree(tree{3});
        y = f(a,b);
    end
end

function out = crossover(t1, t2)
    % 简单随机替换某个子树
    if rand < 0.5 && iscell(t1) && iscell(t2)
        out = {t1{1}, crossover(t1{2}, t2{2}), crossover(t1{3}, t2{3})};
    else
        out = t2;
    end
end

function out = mutate(tree, funcs, terms, depth)
    if rand < 0.1
        out = random_tree(funcs, terms, depth);
    elseif iscell(tree)
        out = {tree{1}, mutate(tree{2}, funcs, terms, depth-1), mutate(tree{3}, funcs, terms, depth-1)};
    else
        out = tree;
    end
end

function best = tournament_selection(pop, fitness)
    k = 3;
    idxs = randi(length(pop), [k,1]);
    [~, best_idx] = min(fitness(idxs));
    best = pop{idxs(best_idx)};
end

function pretty_print_tree(tree, indent)
    if nargin < 2
        indent = '';
    end
    if isa(tree, 'function_handle')
        disp([indent func2str(tree)]);
    elseif iscell(tree)
        disp([indent func2str(tree{1})]);
        pretty_print_tree(tree{2}, [indent '  ']);
        pretty_print_tree(tree{3}, [indent '  ']);
    end
end
