function [M, Y, Z] = plantedsubspace(n, m, p, q, sigmasq)
% n is the dimension of ambient space;
% m is number of points in the ambient space;
% p is the dimension of the subspace we are searching for;
% p < n, p < m;
% q is the fraction of outliers;
% the points will deviate from the planted and outlier subspaces with 
% a covariance sigmasq*Id;

    % generate a planted subspace S and an outlier subspace Sout, 
    % specified by ONBs Y and Z in St(n,p), respectively;
    Y = qr_unique(randn(n, p));
    Z = qr_unique(randn(n, p));
    
    % generate m points M = [z1, ..., zm]
    numoutliers = round(q * m);
    numplanted = m - numoutliers;
    Mplanted = Y * mvnrnd(zeros(p, 1), eye(p), numplanted)';
    Moutliers = Z * mvnrnd(zeros(p, 1), eye(p), numoutliers)';
    M = [Mplanted Moutliers];
    
    % perturb the points z1, ..., zm by a gaussian
    M = M + mvnrnd(zeros(n, 1), sigmasq*eye(n), m)';