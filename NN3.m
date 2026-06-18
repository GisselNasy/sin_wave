function [u, h2, H2, W2, B2, W3, B3] = NN3(w, H1, nn, sw)
    % weights and biasses
    W2 = reshape(w(sw(3,1):sw(3,2)), nn(2)+1, nn(3));
    B2 = reshape(w(sw(4,1):sw(4,2)), 1, nn(3));
    W3 = reshape(w(sw(5,1):sw(5,2)), nn(3), nn(4));
    B3 = w(sw(6,1):sw(6,2));
    
    % u
    h2 = H1*W2 + B2;
    H2 = g(h2);
    u  = H2*W3 + B3;
end