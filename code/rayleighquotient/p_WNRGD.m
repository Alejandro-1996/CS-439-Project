function [stepsizes, cost, gradnorm, x] = p_WNRGD(A, p, b, x, d, gradient_noise_variance, maxiter)
    % run gradient descent with step size eta
    stepsizes = [1/b];
    cost = [Fcn(x, A)];
    next_gradient = gradFcn(x, A, gradient_noise_variance);
    gradnorm = [sqrt(sum(next_gradient.^2))];
    for iter = 2:maxiter
        x = Retr(x, -1/b*next_gradient);

        next_gradient = gradFcn(x, A, gradient_noise_variance);

        b = b + (sum(next_gradient.^2))^(1/p)/b;

        % store step sizes, cost, gradnorm
        stepsizes(iter) = 1/b;
        cost(iter) = Fcn(x, A);
        gradnorm(iter) = sqrt(sum(next_gradient.^2));
    end