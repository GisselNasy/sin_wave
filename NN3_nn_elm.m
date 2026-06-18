function [u, h2, H2, W2, B2, W3, B3] = NN3_nn_elm(w, H1, nn_elm, sw_nn_elm)
    % weights and biasses
    W2 = reshape(w(sw_nn_elm(3,1):sw_nn_elm(3,2)), nn_elm(2)+1, nn_elm(3));
    B2 = reshape(w(sw_nn_elm(4,1):sw_nn_elm(4,2)), 1, nn_elm(3));
    W3 = reshape(w(sw_nn_elm(5,1):sw_nn_elm(5,2)), nn_elm(3), nn_elm(4));
    B3 = w(sw_nn_elm(6,1):sw_nn_elm(6,2));
    
    % u
    h2 = H1*W2 + B2;
    H2 = g(h2);
    u  = H2*W3 + B3;
end