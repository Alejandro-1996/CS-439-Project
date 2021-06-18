function M = grassmannprojectorsfactory(n, p)
% Quick code, NB, April 26, 2021
%
%  Points are orthogonal projectors X of size n x n with rank p.
%
%  They are represented as n x p matrices with orthonormal columns.
%
%  Tangent vectors are represented as n x p matrices orthogonal to the base
%  point.
%
%  Ambient vectors are represented as function handles for matrix-product
%  computation: W takes as input a matrix of size nxq and outputs a matrix
%  of size nxq corresponding to the matrix product.
%

    assert(n >= p, ...
           ['The dimension n of the ambient space must be larger ' ...
            'than the dimension p of the subspaces.']);
    
    M.name = @() sprintf('Grassmann manifold Gr(%d, %d) as projectors', n, p);
    
    M.dim = @() p*(n-p);
    
    M.inner = @(Y, Ydot1, Ydot2) 2*Ydot1(:).'*Ydot2(:);
    
    M.norm = @(Y, Ydot) sqrt(2)*norm(Ydot(:));
    
    M.typicaldist = @() sqrt(2*p);
    
    M.proj = @projection;
    function Yp = projection(Y, W)
        WY = W(Y);
        Yp = WY - Y*multisym(Y'*WY);
    end
    
    M.tangent = @(Y, Ydot) Ydot - Y*(Y'*Ydot);
    
    M.tangent2ambient_is_identity = false;
    M.tangent2ambient = @(Y, Ydot) @(Z) Ydot*(Y'*Z) + Y*(Ydot'*Z);
    
    M.egrad2rgrad = M.proj;
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(Y, egrad, ehess, Ydot)
        
        ehessY = ehess(Y);
        egradY = egrad(Y);
        egradYdot = egrad(Ydot);
        
        ZY = ehessY + egradYdot - 2*Y*(Y'*egradYdot) - Y*(Ydot'*egradY) - Ydot*(Y'*egradY);
        
        rhess = ZY - Y*multisym(Y'*ZY);
    end
    
    M.retr = @retraction;
    function Y = retraction(Y, Ydot, t)
        if nargin < 3
            Y = Y + Ydot;
        else
            Y = Y + t*Ydot;
        end
        % Compute the polar factor
        [u, s, v] = svd(Y, 'econ'); %#ok
        Y = u*v'; % Y (Y^T Y)^(-1/2) = u v', amazing!
    end
    
    M.hash = @(Y) ['z' hashmd5(Y(:))];
    
    M.rand = @random;
    function Y = random()
        [Y, ~] = qr(randn(n, p), 0); % 0 specifies economy size
    end
    
    M.randvec = @randomvec;
    function Ydot = randomvec(Y)
        Ydot = randn(n, p);
        Ydot = Ydot - Y*(Y'*Ydot); % tangentialize
        Ydot = Ydot / (norm(Ydot(:)) * sqrt(2));
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(n, p);
    
    M.transp = @(Y1, Y2, Ydot) Ydot - Y2*(Y2'*Ydot);
    
    M.vec = @(x, u_mat) sqrt(2)*u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec/sqrt(2), [n, p]);
    M.vecmatareisometries = @() true;

end
