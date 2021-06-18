% files modified or created to implement our experiments:
% ./main.m
% ./noisyhubercost.m
% ./noisyhubergrad.m
% ./manopt/manopt/solvers/sgd.m
% ./manopt/manopt/solvers/wnrgd.m
% ./manopt/manopt/solvers/wnrgdmod.m

clear all; close all; clc;

% n is the dimension of ambient space;
% m is number of points in the ambient space;
% p is the dimension of the subspace we are searching for;
% p < n, p < m;
% q is the fraction of outliers;
% the points will deviate from the planted and outlier subspaces with 
% a covariance sigmasq*Id;
n = 200; %3000
m = 2000; %100
p = 10; %5
q = 0.3;
eps = 0.000001;
sigmasq = 0.001;
[M, Y, Z] = plantedsubspace(n, m, p, q, sigmasq);

% solve
manifold = grassmannprojectorsfactory(n, p);

problemHuber.M = manifold;
problemHubernoisy.M = manifold;

problemHuber.cost = @(Y) hubercost(Y, M, eps);
problemHuber.egrad = @(Y) hubergrad(Y, M, eps);
problemHuber.ehess = @(Y, Ydot) huberhess(Y, Ydot, M, eps);
problemHubernoisy.cost = @(Y) hubercost(Y, M, eps);
problemHubernoisy.egrad = @(Y) noisyhubergrad(Y, M, eps);

problemHuber.mat = M;
problemHuber.eps = eps;
problemHubernoisy.mat = M;
problemHubernoisy.eps = eps;

Y1 = manifold.rand();
Ydot = manifold.randvec(Y1);
Y2 = manifold.randvec(Ydot);
binv = manifold.norm(Y1, Ydot) / ...
        manifold.norm(Y1, getGradient(problemHuber, Y1) - ...
        manifold.transp(Y1, Y2, getGradient(problemHuber, Y2)));

[~, infoSGD] = sgd(problemHubernoisy, binv/10);

[~, infoWNGrad] = wnrgd(problemHubernoisy, binv);

[~, infoWNGradmod] = wnrgdmod(problemHubernoisy, binv);

[~, ~, infoFullGD] = steepestdescent(problemHuber);

[~, ~, infoRTR] = trustregions(problemHuber);

% plot Huber cost and grad norm
loglog([infoSGD.time], [infoSGD.cost])
hold on
loglog([infoWNGrad.time], [infoWNGrad.cost])
hold on
loglog([infoWNGradmod.time], [infoWNGradmod.cost])
hold on
loglog([infoFullGD.time], [infoFullGD.cost])
hold on
loglog([infoRTR.time], [infoRTR.cost])
hold off
legend({'SGD', 'WNGrad', 'WNGrad-mod', 'RGD', 'RTR'})
title('cost vs time')