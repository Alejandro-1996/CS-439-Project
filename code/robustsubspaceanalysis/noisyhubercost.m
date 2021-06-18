function val = noisyhubercost(Y, M, eps)
    % return huber cost at X = Y*Y'
    
    sz = size(M);
    m = sz(2);

    i=randi([1 m]);
    zi = M(:,i);
    residual = zi - Y*(Y'*zi);
    normsq = sum(residual.^2);
    val = m * sqrt(normsq + eps^2);
end

