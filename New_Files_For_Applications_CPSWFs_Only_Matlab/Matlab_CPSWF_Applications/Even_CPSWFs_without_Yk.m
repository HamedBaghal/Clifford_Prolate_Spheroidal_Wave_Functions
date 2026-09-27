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
% Compute the radial part of an EVEN Clifford prolate spheroidal wave
% function (the angular monogenic Y_k is omitted).
%
% INPUTS
%   r : radius/radii in [0,1]
%   k : homogeneity degree
%   c : CPSWF bandwidth
%   m : Clifford-Legendre/Galerkin truncation size
%   n : CPSWF radial order, n = 1,...,m
%
% NUMERICAL UPDATE
% This version uses:
%   - correctly sorted CPSWF eigenvectors from
%       Even_CPSWFs_Coefficient.m
%   - a stable Jacobi recurrence for ALL Clifford-Legendre basis
%       functions at once.
%
% It avoids the unstable gamma/binomial alternating sums used previously.
%
% The function name and calling syntax are unchanged.

function G = Even_CPSWFs_without_Yk(r,k,c,m,n)

coeff = Even_CPSWFs_Coefficient(k,c,m,n);

original_size = size(r);
rr = r(:);

% Build C_{0,k}, C_{2,k}, ..., C_{2(m-1),k} stably.
CL = even_clp_table(rr,m-1,k);

G = CL * coeff;
G = reshape(G,original_size);

end


function CL = even_clp_table(r,Nmax,k)
% Return normalized even Clifford-Legendre radial factors
% for N = 0,...,Nmax as columns.

z = 1 - 2*r.^2;

alpha = k;
beta  = 0;

P = zeros(numel(r),Nmax+1);
P(:,1) = 1;

if Nmax >= 1
    P(:,2) = 0.5*((alpha-beta) + (alpha+beta+2).*z);
end

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

N = 0:Nmax;
scale = sqrt(2*k + 4*N + 2)/sqrt(2*pi);

CL = (r.^k).*P;
CL = CL.*scale;

end
