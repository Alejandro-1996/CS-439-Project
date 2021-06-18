function gradf = gradFcn(x, A, gradient_noise_variance)
   gradf = 2*(A*x - (x'*A*x)*x);
   sz = size(gradf);
   euc_noise = gradient_noise_variance * randn(sz(1), 1);
   projected_noise = (euc_noise - (x'*euc_noise)*x);
   gradf = gradf + projected_noise;
end