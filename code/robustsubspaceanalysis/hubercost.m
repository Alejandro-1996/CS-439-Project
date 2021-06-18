function val = hubercost(Y, M, eps)
    % return huber cost at X = Y*Y'
    
    sz = size(M);
    m = sz(2);
    val = 0;
    for i=1:m
        zi = M(:,i);
        residual = zi - Y*(Y'*zi);
        normsq = sum(residual.^2);
        val = val + sqrt(normsq + eps^2);
    end
end

