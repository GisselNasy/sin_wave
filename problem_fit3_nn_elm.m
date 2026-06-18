function [L, G] = problem_fit3_nn_elm(w, H1, u0, nn_elm, sw_nn_elm)
    
    % Neural Network
    [u, h2, H2, W2, B2, W3, B3] = NN3_nn_elm(w, H1, nn_elm, sw_nn_elm);

    % System
    F = u - u0;
    
    % Sum Squared Error
    L = sum(F.^2)/2;

    if nargout > 1
        %% Jacobian

        J = zeros(size(F,1), size(w,2));
        
		dgh2 = dg(h2);
        dgh2W3 = dgh2.*W3';
        

        %% First Hidden Layer --> Second Hidden Layer
        
        % Weight
        n1 = size(W2,1);
        for i = 1:n1
            du = H1(:,i).*dgh2W3;
            J(:, sw_nn_elm(2,2)+i:n1:sw_nn_elm(3,2)) = du;
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