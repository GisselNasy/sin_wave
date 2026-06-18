function [L, G] = problem_fit2_nn_elm(w, x, u0, nn_elm, sw_nn_elm)
    % Neural Network
    [u, h1, H1, h2, H2, W1, B1, W2, B2, W3, B3] = NN_nn_elm(w, x, nn_elm, sw_nn_elm);

    % System
    F = u - u0;
    
    % Sum Squared Error
    L = sum(F.^2)/2;

    if nargout > 1
        %% Jacobian

        J = zeros(size(F,1), size(w,2));
        
        X = [x];
        %% Perbaikan Perhitungan Jacobian
        dgh1 = dg(h1);
        dgh2 = dg(h2);
        
        dgh2W3 = dgh2 .* W3';
        bp1 = dgh2W3 * W2';             % bp1 = backprop ke hidden layer 1
        dgh2W3W2_1 = bp1(:, 2:end);     % dgh2W3W2_1 = input error untuk neuron di Layer 1
        dgh1dgh2W3W2 = dgh1 .* dgh2W3W2_1;
                
        %% Input --> First Hidden Layer

        % Weight
        n1 = size(W1,1);
        for i = 1:n1
            du = X(:,i).*dgh1dgh2W3W2;
            J(:, i:n1:sw_nn_elm(1,2)) = du;
        end

        % Bias
        du = dgh1dgh2W3W2;
        J(:, sw_nn_elm(2,1):sw_nn_elm(2,2)) = du;

        %% First Hidden Layer --> Second Hidden Layer (W2)
        n2 = size(W2,1); 
        for i = 1:n2
            du = H1(:,i) .* dgh2W3; 
            J(:, sw_nn_elm(3,1)+(i-1):n2:sw_nn_elm(3,2)) = du;
        end

        % Bias
        du = dgh2W3; 
        J(:, sw_nn_elm(4,1):sw_nn_elm(4,2)) = du;
        %% Second Hidden Layer --> Output

        % Weight
        du = H2;
        J(:, sw_nn_elm(5,1):sw_nn_elm(5,2)) = du;

        % Bias
        du = ones(size(u));
        J(:, sw_nn_elm(6,1):sw_nn_elm(6,2)) = du;

        %% Gradient
        % Compute the gradient
        G = J' * F;
    end
end