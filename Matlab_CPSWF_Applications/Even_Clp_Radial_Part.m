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
% Computes the RADIAL polynomial factor of a normalized EVEN
% Clifford-Legendre polynomial on the unit disc.
%
% INPUTS
%   r : radius/radii in [0,1]
%   N : radial Clifford-Legendre index N = 0,1,2,...
%   k : homogeneity degree k = 0,1,2,...
%
% STABLE FORMULA
% The old gamma/nchoosek representation is mathematically correct but can
% become numerically unstable at large N or k.  The same radial polynomial
% is
%
%   R_{N,k}(r) =
%      sqrt(2*k+4*N+2) * P_N^(k,0)(1-2*r^2).
%
% Relation with Even_Clp_without_Yk:
%
%   Even_Clp_without_Yk(r,N,k)
%      = r^k/sqrt(2*pi) * Even_Clp_Radial_Part(r,N,k).
%
% The function name and calling syntax are unchanged.

function D = Even_Clp_Radial_Part(r,N,k)

validateattributes(N,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(k,{'numeric'},{'scalar','integer','nonnegative'});

original_size = size(r);
rr = r(:);

PN = jacobi_k0_stable(1 - 2*rr.^2,N,k);

D = sqrt(2*k + 4*N + 2) .* PN;
D = reshape(D,original_size);

end


function PN = jacobi_k0_stable(z,N,k)
% P_N^(k,0)(z), evaluated by a three-term Jacobi recurrence.

alpha = k;
beta  = 0;

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
