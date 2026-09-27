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
% Computes the normalized ODD Clifford-Legendre term with the angular
% Clifford factor omitted.
%
% Your 2-D odd angular convention is
%
%       x Y_k(x)
%
% which, on the unit circle, has components
%
%       -cos((k+1)*theta),   -sin((k+1)*theta).
%
% This function does NOT apply those angular factors.  It preserves the
% sign convention of your original Odd_Clp_without_Yk.m.
%
% INPUTS
%   r : radius/radii in [0,1]
%   N : odd Clifford-Legendre radial index N = 0,1,2,...
%   k : homogeneity degree k = 0,1,2,...
%
% STABLE FORMULA
% --------------
% Your original gamma/binomial formula is exactly equivalent to
%
%   C^odd_{N,k}(r)
%      = -sqrt(2*k+4*N+4)/sqrt(2*pi)
%        * r^(k+1) * P_N^(k+1,0)(1-2*r^2).
%
% The Jacobi representation is evaluated by a stable three-term recurrence
% to avoid catastrophic cancellation at high N and k.
%
% Function name and calling syntax are unchanged.

function D = Odd_Clp_without_Yk(r,N,k)

validateattributes(N,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(k,{'numeric'},{'scalar','integer','nonnegative'});

original_size = size(r);
rr = r(:);

PN = jacobi_alpha0_stable(1 - 2*rr.^2,N,k+1);

D = -sqrt(2*k + 4*N + 4)/sqrt(2*pi) ...
    .* (rr.^(k+1)) .* PN;

D = reshape(D,original_size);

end


function PN = jacobi_alpha0_stable(z,N,alpha)
% P_N^(alpha,0)(z), stable three-term recurrence.

beta = 0;

Pnm1 = ones(size(z));

if N == 0
    PN = Pnm1;
    return
end

Pn = 0.5*((alpha-beta) + (alpha+beta+2).*z);

if N == 1
    PN = Pn;
    return
end

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
