% References
%
% 1. Ghaffari, H. B., Hogan, J. A., & Lakey, J. D. (2022).
%    Properties of Clifford-Legendre Polynomials.
%    Advances in Applied Clifford Algebras, 32(1), 1-25.
%    https://doi.org/10.1007/s00006-021-01179-8
%
% 2. H. Baghal Ghaffari, "Higher-dimensional prolate spheroidal wave
%    functions," Ph.D. dissertation, The University of Newcastle, 2022.
%
% STABLE EVEN-CPSWF FINITE-FOURIER EIGENVALUE
% -------------------------------------------
%
% Calling syntax is unchanged:
%
%       q = Even_Eigenvalue_CPSWFs(k,c,m,n)
%
% IMPORTANT:
% This version uses the FINITE HANKEL TRANSFORM WITH THE NORMALIZATION
% FROM THE CPSWF THEORY, not the ordinary-PSWF Hankel normalization.
%
% In dimension m=2, Proposition 3.3 gives
%
%       gamma_N^(k) = (-i)^k * sqrt(c) * mu_(2N)^(k,c),
%
% where gamma_N^(k) is the eigenvalue of
%
%       H_c^(k) f(s)
%         = sqrt(2*pi) * integral_0^1
%             sqrt(2*pi*c*s*r) J_k(2*pi*c*s*r) f(r) dr.
%
% Therefore
%
%       mu_(2N)^(k,c)
%         = i^k * gamma_N^(k) / sqrt(c).
%
% The square-root weighted radial eigenfunction is
%
%       S_N(r) = r^(k+1/2) P_N^(k,c)(r^2).
%
% We compute gamma directly from the above integral equation at the
% quadrature point where |S_N| is largest.  This avoids the numerically
% sensitive endpoint quotient used in the original implementation.
%
% REQUIRED FILES:
%       Even_CPSWFs_Matrix.m
%       lgwt.m
%
% The function caches the complete eigenvalue vector for the most recently
% requested (k,c,m), so calls with n=1,2,... are fast after the first one.

function q = Even_Eigenvalue_CPSWFs(k,c,m,n)

validateattributes(k,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(c,{'numeric'},{'scalar','real','positive'});
validateattributes(m,{'numeric'},{'scalar','integer','positive'});
validateattributes(n,{'numeric'},{'scalar','integer','>=',1,'<=',m});

if isempty(which('Even_CPSWFs_Matrix'))
    error('Even_CPSWFs_Matrix.m was not found on the MATLAB path.');
end

if isempty(which('lgwt'))
    error('lgwt.m was not found on the MATLAB path.');
end

persistent cache_k cache_c cache_m cache_q

same_cache = ~isempty(cache_q) && ...
             isequal(cache_k,k) && ...
             isequal(cache_c,c) && ...
             isequal(cache_m,m);

if ~same_cache

    %% 1. Differential-operator eigenvectors

    A = Even_CPSWFs_Matrix(k,c,m);
    A = 0.5*(A + A.');

    [V,D] = eig(A);

    [~,idx] = sort(real(diag(D)),'ascend');
    V = real(V(:,idx));

    % Deterministic signs only; eigenvalue quotient itself is sign-invariant.
    for j = 1:m
        [~,imax] = max(abs(V(:,j)));
        if V(imax,j) < 0
            V(:,j) = -V(:,j);
        end
    end

    %% 2. Accurate Gauss-Legendre quadrature on [0,1]

    quadN = max(500,4*m+100);

    [rq,wq] = lgwt(quadN,0,1);

    rq = rq(:);
    wq = wq(:);

    %% 3. Construct S_N(r) = r^(k+1/2) P_N^(k,c)(r^2)
    %
    % The normalized radial Clifford-Legendre basis is
    %
    %   R_{j,k}(r)
    %     = sqrt(2*k+4*j+2) P_j^(k,0)(1-2r^2).
    %
    % A common normalization factor does not affect the eigenvalue quotient.

    Jac = local_jacobi_alpha0_table(1-2*rq.^2,m-1,k);

    j = 0:(m-1);
    scale = sqrt(2*k + 4*j + 2);

    radial_basis = Jac .* scale;

    S = ((rq.^(k+0.5)) .* radial_basis) * V;

    %% 4. Finite-Hankel eigenvalues gamma and Clifford-FT eigenvalues mu

    q_all = zeros(1,m);

    for jj = 1:m

        Sj = S(:,jj);

        % Evaluate the eigen-equation where the eigenfunction is largest.
        [~,imax] = max(abs(Sj));

        s = rq(imax);
        denom = Sj(imax);

        if ~isfinite(denom) || abs(denom) < 100*eps
            error(['Unable to compute eigenvalue stably for ', ...
                   'k=%d, n=%d.'],k,jj);
        end

        % EXACT finite-Hankel normalization from Definition 3.2:
        %
        % sqrt(2*pi)*sqrt(2*pi*c*s*r)*J_k(2*pi*c*s*r).
        kernel = sqrt(2*pi) .* ...
                 sqrt(2*pi*c*s*rq) .* ...
                 besselj(k,2*pi*c*s*rq);

        gamma_hankel = sum(wq .* kernel .* Sj) / denom;

        % Proposition 3.3, m=2:
        %
        % gamma = (-i)^k sqrt(c) mu
        % => mu = i^k gamma/sqrt(c).
        mu = (1i)^k * gamma_hankel / sqrt(c);

        q_all(jj) = mu;
    end

    cache_k = k;
    cache_c = c;
    cache_m = m;
    cache_q = q_all;
end

q = cache_q(n);

end


function P = local_jacobi_alpha0_table(z,Nmax,alpha)
% Stable evaluation of
%
%   P_0^(alpha,0)(z), ..., P_Nmax^(alpha,0)(z).

z = z(:);
beta = 0;

P = zeros(numel(z),Nmax+1);
P(:,1) = 1;

if Nmax == 0
    return
end

P(:,2) = 0.5*((alpha-beta) + (alpha+beta+2).*z);

for n = 1:Nmax-1

    A1 = 2*(n+1)*(n+alpha+beta+1)*(2*n+alpha+beta);

    A2 = (2*n+alpha+beta+1)*(alpha^2-beta^2);

    A3 = (2*n+alpha+beta) * ...
         (2*n+alpha+beta+1) * ...
         (2*n+alpha+beta+2);

    A4 = 2*(n+alpha)*(n+beta)*(2*n+alpha+beta+2);

    P(:,n+2) = ...
        ((A2 + A3.*z).*P(:,n+1) - A4.*P(:,n))./A1;
end

end
