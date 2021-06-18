function fcthandle = hubergrad(Y, M, eps)
    % nabla f(X) * Z, where X = Y*Y'
    
    sz = size(M);
    m = sz(2);
    function out = gradtimesZ(Z)
        % Z is nxp
        out = zeros(size(Z));
        for i=1:m
            zi = M(:,i);
            residual = zi - Y*(Y'*zi);
            normsq = sum(residual.^2);
            zitimesZ = zi*(zi'*Z);
            out = out + 0.5/sqrt(normsq+eps^2) * ...
                ((Y*(Y'*zitimesZ) - zitimesZ) + ((zi*(zi'*Y))*(Y'*Z) - zitimesZ));
        end
    end
    
    fcthandle = @gradtimesZ;
end

