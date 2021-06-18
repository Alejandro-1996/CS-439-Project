function fcthandle = noisyhubergrad(Y, M, eps)
    % nabla f(X) * Z, where X = Y*Y'
    
    sz = size(M);
    m = sz(2);
    function out = gradtimesZ(Z)
        minibatchsize = round(m/100);
        
        % Z is nxp
        out = zeros(size(Z));
        for i=randi([1 m], 1, minibatchsize)
            zi = M(:,i);
            residual = zi - Y*(Y'*zi);
            normsq = sum(residual.^2);
            zitimesZ = zi*(zi'*Z);
            out = out + (m / minibatchsize) * 0.5/sqrt(normsq+eps^2) * ...
                ((Y*(Y'*zitimesZ) - zitimesZ) + ((zi*(zi'*Y))*(Y'*Z) - zitimesZ));
        end
    end
    
    fcthandle = @gradtimesZ;
end

