function [L, G] = problem_fit3(w, H1, u0, nn, sw)
    
    % Neural Network
    [u, h2, H2, W2, B2, W3, B3] = NN3(w, H1, nn, sw);

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
            J(:, sw(2,2)+i:n1:sw(3,2)) = du;
        end

        % Bias
        du = dgh2W3;
        J(:, sw(4,1):sw(4,2)) = du;

        %% Second Hidden Layer --> Output

        % Weight
        du = H2;
        J(:, sw(5,1):sw(5,2)) = du;

        % Bias
        du = ones(size(u));
        J(:, sw(6,1):sw(6,2)) = du;

        %% Gradient
        % Compute the gradient
        G = J' * F;
    end
end