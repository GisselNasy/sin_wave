function [W1, B1, W2, B2, W3, B3, h1, H1, h2, H2, u] = NN2(w, X)
    W1 = reshape(w(1:100), 1, 100); % 100
    B1 = reshape(w(101:200), 1, 100); % 100 
    W2 = reshape(w(201:10200), 100, 100); % 10000
    B2 = reshape(w(10201:10300), 1, 100); % 100
    W3 = reshape(w(10301:10400), 100, 1); % 100
    B3 = w(10401); % 1

    h1 = X*W1 + B1;
    H1 = tanh(h1);
    h2 = H1*W2 + B2;
    H2 = tanh(h2);
    u = (H2*W3 + B3);
end