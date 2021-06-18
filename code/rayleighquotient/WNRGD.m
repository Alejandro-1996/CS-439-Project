% p=2 seems best over a wide range of noises
gradient_noise_variance = 0.01; % 0.01, 0.1, 1, 10
d = 100;
maxiter = 320000;
A = randn(d, d);
A = A + A';
iilist = [0, 1, 2, 3];
add = 1;
legendTemplate = {'p=1', 'p=2', 'p=4', 'p=8'};
sz = size(iilist);

% initial point
x = zeros(d, 1);
x(1) = 1;

% initialize b
perturbation = 0.01 * randn(d, 1);
y = x + perturbation;
y = y / sqrt(sum(y.^2));
graddiff = gradFcn(x, A, gradient_noise_variance) - gradFcn(x, A, gradient_noise_variance);
b = acos(y' * x) / sqrt(sum(graddiff.^2));

stepsizesS = cell(sz(2), 1);
costS = cell(sz(2), 1);
gradnormS = cell(sz(2), 1);
xS = cell(sz(2), 1);
for ii = iilist
    p = 2^ii;
    [stepsizes, cost, gradnorm, x] = p_WNRGD(A, p, b, x, d, gradient_noise_variance, maxiter);
    stepsizesS{ii+add} = stepsizes;
    costS{ii+add} = cost;
    gradnormS{ii+add} = gradnorm;
    xS{ii+add} = x;
end

[V, D] = eig(A);
minV = V(:, 1);
minV = minV / sqrt(sum(minV.^2));

tiledlayout(2,2)

nexttile
for ii = iilist
    loglog(stepsizesS{ii+add})
    hold on
end
hold off
title('stepsizes')
legend(legendTemplate)

nexttile
for ii = iilist
    loglog(costS{ii+add} - Fcn(minV, A) + 0.01)
    hold on
end
hold off
title('cost')
legend(legendTemplate)

nexttile
for ii = iilist
    loglog(gradnormS{ii+add})
    hold on
end
hold off
title('gradnorm')
legend(legendTemplate)

nexttile
histogram(eig(A))