
%% 参数设定
r     = 0.05;    % 无风险利率
tau   = 0.2;     % 公司税率
delta = 0.1;     % 资本折旧率
alpha = 0.5;     % 投资调整成本系数
b     = 0.03;    % 基础票息率
mu    = 1.0;     % 固有生产率
sigma = 0.2;     % 生产率波动率
k     = 0.1;     % 碳税系数
e     = 0.05;    % 碳排放预期增长率
X_b   = 1;      % 破产阈值

psi_values = [0.001, 0.1];   % 要对比的 ψ 值
colors     = {'b-','m--'};   % 绘图样式

%% 状态空间离散化
N     = 1000;
X_max = 1000;                 % 适度增大上限以更好近似 ∞
X_grid = linspace(X_b, X_max, N)';
dX      = X_grid(2) - X_grid(1);

%% 预分配结果矩阵
E_all     = zeros(N, numel(psi_values));
i_opt_all = zeros(N, numel(psi_values));

%% main循环：对每个 ψ 求解 HJB
max_iter = 7000;
tol      = 1e-6;

for p = 1:numel(psi_values)
    psi = psi_values(p);
    
    % 初始化 E 与 i*
    E         = zeros(N,1);
    i_opt     = zeros(N,1);
    E(1)      = 0;
    E(end)    = 2 * X_grid(end);

    % 迭代求解 HJB
    for iter = 1:max_iter
        E_old = E;
        
        for j = 2:N-1
            X = X_grid(j);
            % 中心差分计算一阶、二阶导数
            dEdX   = (E(j+1) - E(j-1)) / (2*dX);
            d2EdX2 = (E(j+1) - 2*E(j) + E(j-1)) / (dX^2);
            
            % HJB 
            H = @(i) (1-tau)*(mu*(i*X - delta*X) - b*(1 - psi*e*X)) ...
                     - k*e*X - 0.5*alpha*X*i^2 ...
                     + dEdX*(i*X - delta*X) ...
                     + 0.5*d2EdX2*sigma^2*X^2;
                 
            % 寻找最优投资 i*（最大化 H）
            [i_star, Hval] = fminbnd(@(i) -H(i), 0, 0.5);
            
            % 用 H / r 更新 E，后续使用可视化Ex
            E(j)       = Hval / r;
            i_opt(j)   = i_star;
        end
        
        % 重设边界条件
        E(1)      = 0;
        E(end)    = 2 * X_grid(end);

        % 收敛检测
        if max(abs(E - E_old)) < tol
            fprintf('ψ = %.4f: Converged at iter = %d\n', psi, iter);
            break;
        end
    end
    
    % 存储
    E_all(:,p)     = E;
    i_opt_all(:,p) = i_opt;
end

%% 1可视化：X vs i(X) 对比
figure('Position',[100,100,700,400]);
hold on;
for p = 1:numel(psi_values)
    plot(X_grid, i_opt_all(:,p), colors{p}, 'LineWidth', 2);
end
xlabel('Capital Stock $X_t$', 'Interpreter','latex');
ylabel('Optimal Investment Rate $i^*(X_t)$', 'Interpreter','latex');
title('Optimal Investment Policy under Different $\psi$', 'Interpreter','latex');
legend_str = arrayfun(@(v) sprintf('\\psi=%.3f',v), psi_values, 'UniformOutput',false);
legend(legend_str, 'Location','northeast','Interpreter','latex');
grid on;
xlim([X_b, 10]);  % 关注较低 X 区间以便观察差异
hold off;
