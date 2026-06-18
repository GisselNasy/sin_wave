
%% Check Gradient

w = w0;

options = optimoptions('fminunc', ...
    'Display', 'iter', ...
    'HessianApproximation', 'bfgs', ...
    'MaxFunctionEvaluations', 1e+20, ...
    'MaxIterations', 0, ...
    'SpecifyObjectiveGradient', false, ...
    'StepTolerance', 1e-12);

tic
[w, fval, exitflag, output, grad1] = fminunc(@problem_fit2, w, options, x, u0, nn, sw);
toc

[L, G] = problem_fit2(w, x, u0, nn, sw)

dgrad1 = abs(grad1-G)
dgrad2 = max(abs(grad1-G))

Stop