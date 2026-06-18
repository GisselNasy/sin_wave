function [u, h1, H1, h2, H2, W1, B1, W2, B2, W3, B3] = NN_nn_elm(w, x, nn_elm, sw_nn_elm)
    % W1 = reshape(w(1:100), 1, 100); % 100
    % B1 = reshape(w(101:200), 1, 100); % 100 
    % W2 = reshape(w(201:10200), 100, 100); % 10000
    % B2 = reshape(w(10201:10300), 1, 100); % 100
    % W3 = reshape(w(10301:10400), 100, 1); % 100
    % B3 = w(10401); % 1

    % W1 = reshape(w(1:5), 1, 5); % 5
    % B1 = reshape(w(6:10), 1, 5); % 5 
    % W2 = reshape(w(11:35), 5, 5); % 25
    % B2 = reshape(w(36:40), 1, 5); % 5
    % W3 = reshape(w(41:45), 5, 1); % 5
    % B3 = w(46); % 1

    % W1 = reshape(w(1:10), 1, 10); % 10
    % B1 = reshape(w(11:20), 1, 10); % 10 
    % W2 = reshape(w(21:120), 10, 10); % 100
    % B2 = reshape(w(121:130), 1, 10); % 10
    % W3 = reshape(w(131:140), 10, 1); % 10
    % B3 = w(141); % 1

    % W1 = reshape(w(1:10), 1, 10); % 10
    % B1 = reshape(w(11:20), 1, 10); % 10 
    % W2 = reshape(w(21:220), 10, 20); % 200
    % B2 = reshape(w(221:240), 1, 20); % 20
    % W3 = reshape(w(241:260), 20, 1); % 20
    % B3 = w(261); % 1
    % 
    % H1 = tanh(X *W1 + B1);
    % H2 = tanh(H1*W2 + B2);
    % u = (H2*W3 + B3);

    % weights and biasses
    W1 = reshape(w(sw_nn_elm(1,1):sw_nn_elm(1,2)), nn_elm(1), nn_elm(2));
    B1 = reshape(w(sw_nn_elm(2,1):sw_nn_elm(2,2)), 1, nn_elm(2));
    W2 = reshape(w(sw_nn_elm(3,1):sw_nn_elm(3,2)), nn_elm(2)+1, nn_elm(3));
    B2 = reshape(w(sw_nn_elm(4,1):sw_nn_elm(4,2)), 1, nn_elm(3));
    W3 = reshape(w(sw_nn_elm(5,1):sw_nn_elm(5,2)), nn_elm(3), nn_elm(4));
    B3 = w(sw_nn_elm(6,1):sw_nn_elm(6,2));
    
    % u
    X  = [x];
    h1 = X*W1 + B1;
    H1 = g(h1);
    H1 = [x H1];
    h2 = H1*W2 + B2;
    H2 = g(h2);
    u  = H2*W3 + B3;
end