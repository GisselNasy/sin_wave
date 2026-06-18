function u = d3g(x)
    u = ( 4*((tanh(x)).^2) - 2*((sech(x)).^2) ).*((sech(x)).^2);
end