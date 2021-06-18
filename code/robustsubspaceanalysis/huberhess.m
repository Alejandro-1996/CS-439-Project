function fcthandle = huberhess(Y, Ydot, M, eps)
    % nabla^2 f(X)[Xdot] * Z, where X = Y*Y' and Ydot=(eye(n)-Y*Y')*Xdot*Y
    
    sz = size(M);
    m = sz(2);
    function out = hesstimesZ(Z)
        % Z is nxp
        out = zeros(size(Z));
        for i=1:m
            zi = M(:,i);
            residual = zi - Y*(Y'*zi);
            normsq = sum(residual.^2);
            zitimesZ = zi*(zi'*Z);
            
            % first term
            out = out + 0.5/sqrt(normsq+eps^2) * ((Y*(Ydot'*zitimesZ) + Ydot*(Y'*zitimesZ)) + (zi*((zi'*Ydot)*(Y'*Z)) + zi*((zi'*Y)*(Ydot'*Z))));
            
            % second term
            innerproduct = fasttrace(Y*(Ydot'* zi), zi);
            out = out + 0.5/(normsq+eps^2)^(3/2) * innerproduct ...
                * ((Y*(Y'*zitimesZ) - zitimesZ) + ((zi*(zi'*Y))*(Y'*Z) - zitimesZ));
        end
        
        function tr = fasttrace(vec1, vec2)
            % efficiently computes trace(vec1 * vec2')
            % vec1 and vec2 are both nx1
            szvec = size(vec1);
            n = szvec(1);
            tr = 0;
            for j=1:n
               tr = tr + vec1(j)*vec2(j); 
            end
        end
        
    end
    
    fcthandle = @hesstimesZ;
end

