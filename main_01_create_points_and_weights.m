clc
clear
close all

% Trial
tr = 3;


%% Data

% % NN Weight
% nn = [1 5 92 1]; % neural network architecture
% sw = position_weights(nn);

% % Initialization
% n_weight = sw(6,2); % number of weights and biasses
% w0 = zeros(1, n_weight);
% w0(sw(1,1):sw(1,2)) = (rand(1, sw(1,2)-sw(1,1)+1)*2-1)*sqrt(6/( nn(1) + nn(2) ));
% w0(sw(3,1):sw(3,2)) = (rand(1, sw(3,2)-sw(3,1)+1)*2-1)*sqrt(6/( nn(2)+1 + nn(3) ));
% w0(sw(5,1):sw(5,2)) = (rand(1, sw(5,2)-sw(5,1)+1)*2-1)*sqrt(6/( nn(3) + nn(4) ));
% 
% writematrix(w0, "csv_w0_tr" + tr + "_nn_sinwave_transformer.csv", 'Delimiter', ';');

% NN-ELM weight
nn_elm = [1 40 40 1]; % neural network architecture
sw_nn_elm = position_weights(nn_elm);

% Initialization
n_weight_nn_elm = sw_nn_elm(6,2); % number of weights and biasses
w0_nn_elm = zeros(1, n_weight_nn_elm);
w0_nn_elm(sw_nn_elm(1,1):sw_nn_elm(1,2)) = (rand(1, sw_nn_elm(1,2)-sw_nn_elm(1,1)+1)*2-1)*sqrt(6/( nn_elm(1) + nn_elm(2) ));
w0_nn_elm(sw_nn_elm(3,1):sw_nn_elm(3,2)) = (rand(1, sw_nn_elm(3,2)-sw_nn_elm(3,1)+1)*2-1)*sqrt(6/( nn_elm(2)+1 + nn_elm(3) ));
w0_nn_elm(sw_nn_elm(5,1):sw_nn_elm(5,2)) = (rand(1, sw_nn_elm(5,2)-sw_nn_elm(5,1)+1)*2-1)*sqrt(6/( nn_elm(3) + nn_elm(4) ));

writematrix(w0_nn_elm, "csv_w0_tr" + tr + "_nn_sinwave_transformer.csv", 'Delimiter', ';');

