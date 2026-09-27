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
% This code computes the normalized EVEN Clifford-Legendre polynomial
% radial factor on the unit disc; the angular monogenic Y_k is omitted.
%
% INPUTS
%   r : radius/radii in [0,1] (scalar, vector, or array)
%   N : radial Clifford-Legendre index N = 0,1,2,...
%   k : homogeneity degree k = 0,1,2,...
%
% NUMERICAL UPDATE
% The previous explicit gamma/binomial representation becomes unstable
% for large N and k because very large alternating terms cancel.
% The same polynomial is evaluated here using the Jacobi identity
%
%   C_{2N,k}(r)
%     = sqrt(2*k+4*N+2)/sqrt(2*pi)
%       * r^k * P_N^(k,0)(1-2*r^2),
%
% together with a stable three-term recurrence for Jacobi polynomials.
%
% The function name and calling syntax are unchanged.

function D = Even_Clp_without_Yk(r,N,k)

if ~(isscalar(N) && N >= 0 && N == floor(N))
    error('N must be a nonnegative integer.');
end

if ~(isscalar(k) && k >= 0 && k == floor(k))
    error('k must be a nonnegative integer.');
end

original_size = size(r);
rr = r(:);

z = 1 - 2*rr.^2;

alpha = k;
beta  = 0;

% P_0^(alpha,beta)
Pnm1 = ones(size(rr));

if N == 0
    PN = Pnm1;
else
    % P_1^(alpha,beta)
    Pn = 0.5*((alpha-beta) + (alpha+beta+2).*z);

    if N == 1
        PN = Pn;
    else
        % Compute P_{n+1} from P_n and P_{n-1}.
        for n = 1:N-1

            A1 = 2*(n+1)*(n+alpha+beta+1)*(2*n+alpha+beta);

            A2 = (2*n+alpha+beta+1)*(alpha^2-beta^2);

            A3 = (2*n+alpha+beta) * ...
                 (2*n+alpha+beta+1) * ...
                 (2*n+alpha+beta+2);

            A4 = 2*(n+alpha)*(n+beta)*(2*n+alpha+beta+2);

            Pnp1 = ((A2 + A3.*z).*Pn - A4.*Pnm1)./A1;

            Pnm1 = Pn;
            Pn   = Pnp1;
        end

        PN = Pn;
    end
end

normalization = sqrt(2*k + 4*N + 2)/sqrt(2*pi);

D = normalization .* (rr.^k) .* PN;
D = reshape(D,original_size);

end
