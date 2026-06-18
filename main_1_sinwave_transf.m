clc; clear; close all;

tr = 1;

n = 1000;
n = 500;
x = linspace(-2,2,n+1)';
u0 = sin(2*pi*x)+cos(pi*x.^2)+cos(pi*x.^3).*sin(pi*x.^3);

figure;
plot(x,u0,'k','LineWidth',1.5);
title('Target Function');
grid on;

%  NETWORK STRUCTURE
nn = [1 5 100 1];
sw = position_weights(nn);
n_weight = sw(6,2);

%  INITIALIZATION
w0 = readmatrix('csv_w0_tr2_sinwave_transformer.csv');

global MSE_hist time_hist
MSE_hist  = [];
time_hist = [];

% N = 500;
N = 1000;


%%  NN saja
disp('NN');

%  OPTIM OPTIONS
max_iter = N;

options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'HessianApproximation','bfgs', ...
    'SpecifyObjectiveGradient', true, ...
    'Display','iter', ...
    'MaxIterations', max_iter, ...
    'MaxFunctionEvaluations', 1e10, ...
    'OutputFcn', @(w,optimValues,state) ...
        outfun(w,optimValues,state,x,u0,nn,sw));

tic;
[w_nn,~,~,output_nn] = ...
    fminunc(@(w) problem_fit2(w,x,u0,nn,sw), w0, options);
t_nn = toc;

u_pred_nn = NN(w_nn,x,nn,sw);
mse_nn = MSE_hist(end);
MSE_hist_nn = MSE_hist;
time_hist_nn = time_hist;

fprintf('NN selesai. Iterasi: %d | MSE: %.3e | Runtime: %.2fs\n', ...
        output_nn.iterations, mse_nn, t_nn);

% PLOT
figure;
set(gcf, 'Position', [100, 100, 700, 350]);

subplot(1, 2, 1); 
plot(time_hist_nn, MSE_hist_nn, 'LineWidth', 1.5);
set(gca, 'YScale', 'log');
axis square;
xlabel('Time (s)'); ylabel('MSE');
title('NN Training Progress (Time)');
grid on;

subplot(1, 2, 2); 
plot(0:length(MSE_hist_nn)-1, MSE_hist_nn, 'LineWidth', 1.5);
set(gca, 'YScale', 'log');
axis square;
xlabel('Iteration'); ylabel('MSE');
title('NN Training Progress (Iteration)');
grid on;

figure;
plot(x,u0,'k','LineWidth',2); hold on;
plot(x,u_pred_nn,'r--','LineWidth',1.5);
title('NN (True vs Prediction)');
legend('True','NN murni');
grid on;


%%  NN–ELM 
disp('NN–ELM');

MSE_hist  = [];
time_hist = [];

% --- Phase 1: Train Hidden Layer 1 & 2 ---
max_iter = N/2;
options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'HessianApproximation','bfgs', ...
    'SpecifyObjectiveGradient', true, ...
    'Display','iter', ...
    'MaxIterations', max_iter, ...
    'MaxFunctionEvaluations', 1e10, ...
    'OutputFcn', @(w,optimValues,state) outfun(w,optimValues,state,x,u0,nn,sw));

tic;
[w_stage1,~,~,out1] = fminunc(@(w) problem_fit2(w,x,u0,nn,sw), w0, options);
t_stage1 = toc;
idx_end_p1 = length(MSE_hist);

MSE_phase1 = MSE_hist;
time_p1 = time_hist;

% --- Phase 2: Freeze Hidden Layer 1, train Hidden Layer 2 ---
W1 = reshape(w_stage1(sw(1,1):sw(1,2)), nn(1), nn(2));
B1 = reshape(w_stage1(sw(2,1):sw(2,2)), 1, nn(2));
H1 = g(x*W1 + B1);
H1 = [x H1];
w2 = w_stage1(sw(3,1):sw(6,2));

sw_edit = sw - sw(2,2);
outfun_phase2_handle = @(w,optimValues,state,varargin) outfun_phase2_local(w,optimValues,state,H1,u0,nn,sw_edit,n);

options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'HessianApproximation','bfgs', ...
    'SpecifyObjectiveGradient', true, ...
    'Display','iter', ...
    'MaxIterations', max_iter, ...
    'MaxFunctionEvaluations', 1e10, ...
    'OutputFcn', outfun_phase2_handle);

% [w2,fval,exitflag,out2,grad] = fminunc(@problem_fit3, w2, options, H1, u0, nn, sw_edit);
[w2,~,~,out2] = fminunc(@problem_fit3, w2, options, H1, u0, nn, sw_edit);
t_stage2 = toc - t_stage1;

MSE_phase2 = MSE_hist;

%PLOT
figure;
set(gcf, 'Position', [100, 100, 700, 350]);

subplot(1,2,1);
plot(time_hist(1:idx_end_p1), MSE_hist(1:idx_end_p1));
set(gca, 'YScale', 'log');
axis square; grid on;
title('Phase 1');
ylabel('MSE'); xlabel('Time (s)');

subplot(1, 2, 2);
plot(time_hist(idx_end_p1+1:end), MSE_hist(idx_end_p1+1:end));
set(gca, 'YScale', 'log');
axis square; grid on;
title('Phase 2');
ylabel('MSE'); xlabel('Time (s)');

figure;
plot(time_hist, MSE_hist, 'k', 'LineWidth', 1.5);
set(gca, 'YScale', 'log');
hold on;
xl = xline(t_stage1, '--r', 'LineWidth', 1.5);
xl.LabelVerticalAlignment = 'bottom';
xl.LabelHorizontalAlignment = 'right';
grid on;
title('Phase 1 + Phase 2');
ylabel('MSE'); xlabel('Time (s)');
legend('MSE Curve', 'Transition Point (Phase 1 to 2)');

%  Phase 3: Freeze HL2 + ELM
tic;

% Extract frozen hidden layers
w_stage2 = [w_stage1(sw(1,1):sw(2,2)) w2];
% W1 = reshape(w_stage2(sw(1,1):sw(1,2)), nn(1), nn(2));
% B1 = reshape(w_stage2(sw(2,1):sw(2,2)), 1, nn(2));
W2 = reshape(w_stage2(sw(3,1):sw(3,2)), nn(2)+1, nn(3));
B2 = reshape(w_stage2(sw(4,1):sw(4,2)), 1, nn(3));

% Feature extraction
H2 = g(H1*W2 + B2);
H2_aug = [H2 ones(size(H2,1),1)];

% output ELM
w3 = H2_aug \ u0;

% Final NN–ELM weights
w_nn_elm = [w_stage2(1:sw(4,2)) w3'];
t_elm = toc;

%  Evaluation
u_pred_nn_elm = NN(w_nn_elm, x, nn, sw);
mse_nn_elm = (2/size(x,1))*problem_fit2(w_nn_elm,x,u0,nn,sw);

t_nn_elm = t_stage1 + t_stage2 + t_elm;

%  NN–ELM training curve
MSE_nn_elm_curve = [ ...
    MSE_phase1, ...
    MSE_phase2, ...
    mse_nn_elm ];

fprintf('NN–ELM selesai.\n');
fprintf('  Phase 1 iters: %d\n', out1.iterations);
fprintf('  Phase 2 iters: %d\n', out2.iterations);
fprintf('  Final MSE: %.3e | Time: %.2fs\n', mse_nn_elm, t_nn_elm);


%% ELM saja
disp('ELM');

w = w0;

W1 = reshape(w(sw(1,1):sw(1,2)), nn(1), nn(2));
B1 = reshape(w(sw(2,1):sw(2,2)), 1, nn(2));
W2 = reshape(w(sw(3,1):sw(3,2)), nn(2)+1, nn(3));
B2 = reshape(w(sw(4,1):sw(4,2)), 1, nn(3));

H1 = g(x*W1 + B1);
H1 = [x H1];
H2 = g(H1*W2 + B2);

H2_aug = [H2 ones(size(H2,1),1)];

tic;
w3_elm = H2_aug \ u0;
t_elm = toc;

w_elm = [W1(:); B1(:); W2(:); B2(:); w3_elm];

u_pred_elm = NN(w_elm,x,nn,sw);
mse_elm = (2/size(x,1))*problem_fit2(w_elm,x,u0,nn,sw);

fprintf('ELM selesai. MSE: %.3e | Runtime: %.2fs\n', ...
        mse_elm, t_elm);

%%  COMPARISON
figure;
plot(x,u0,'k','LineWidth',2.5); hold on;
plot(x,u_pred_nn,'r-.','LineWidth',1.5);
plot(x,u_pred_nn_elm,'b--','LineWidth',1.5);
plot(x,u_pred_elm,'g:','LineWidth',1.5);
legend('True','NN','NN–ELM','ELM','Location','best');
xlabel('x'); ylabel('u(x)');
title('Perbandingan Prediksi');
grid on;

figure;
bar([mse_nn,mse_nn_elm,mse_elm]);
set(gca,'XTickLabel',{'NN','NN–ELM','ELM'});
ylabel('MSE');
title('Perbandingan Error');
grid on;

fprintf('\n=== SUMMARY ===\n');
fprintf('NN:        MSE = %.3e | Time = %.2fs\n', mse_nn, t_nn);
fprintf('NN–ELM:    MSE = %.3e | Time = %.2fs\n', mse_nn_elm, t_nn_elm);
fprintf('ELM murni: MSE = %.3e | Time = %.2fs\n', mse_elm, t_elm);

%  TRAINING CURVE
figure;
plot(0:length(MSE_hist_nn)-1, MSE_hist_nn, ...
     'r-', 'LineWidth', 1.8); hold on;

plot(0:length(MSE_nn_elm_curve)-1, MSE_nn_elm_curve, ...
     'b--', 'LineWidth', 1.8);

plot(length(MSE_nn_elm_curve)-1, MSE_nn_elm_curve(end), ...
     'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');

set(gca,'YScale','log');
xlabel('Iteration');
ylabel('MSE');
legend('NN','NN–ELM','Location','best');
title('NN vs NN–ELM Training');
grid on;

%% helper function
function stop = outfun(w, optimValues, state, x, u0, nn, sw, varargin) % Tambahkan varargin
    stop = false;
    global MSE_hist time_hist
    switch state
        case 'init'
            % Ambil hanya output pertama (L) dari problem_fit
            [L, ~] = problem_fit2(w, x, u0, nn, sw);
            MSE_hist = (2/size(x,1)) * L;
            time_hist = toc;
        case 'iter'
            [L, ~] = problem_fit2(w, x, u0, nn, sw);
            MSE_hist(end+1) = (2/size(x,1)) * L;
            time_hist(end+1) = toc;
    end
end

function stop = outfun_phase2_local(w,optimValues,state,H1,u0,nn,sw,n,varargin)
stop = false;

global MSE_hist
global time_hist

switch state
    case 'init'
        MSE_hist(end+1) = (2/n)*problem_fit3(w,H1,u0,nn,sw);
        time_hist(end+1) = toc;

    case 'iter'
        MSE_hist(end+1) = (2/n)*problem_fit3(w,H1,u0,nn,sw);
        time_hist(end+1) = toc;
end
end