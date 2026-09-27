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
% Computes the RADIAL polynomial part of an EVEN CPSWF.
%
% INPUTS
%   r : radius/radii in [0,1]
%   k : homogeneity degree
%   c : CPSWF bandwidth
%   m : Clifford-Legendre/Galerkin truncation size
%   n : CPSWF radial order, n = 1,...,m
%
% STABILITY UPDATE
% The previous implementation summed Even_Clp_Radial_Part terms whose
% explicit gamma/binomial formulas could become unstable.  This version
% builds the whole radial Clifford-Legendre table with a Jacobi recurrence
% and performs one matrix-vector product.
%
% The function name and calling syntax are unchanged.

function G = Even_CPSWFs_Radial_Part(r,k,c,m,n)

coeff = Even_CPSWFs_Coefficient(k,c,m,n);

original_size = size(r);
rr = r(:);

RCL = even_clp_radial_table(rr,m-1,k);

G = RCL * coeff;
G = reshape(G,original_size);

end


function RCL = even_clp_radial_table(r,Nmax,k)
% Columns are Even_Clp_Radial_Part(r,N,k), N=0,...,Nmax:
%
%   sqrt(2*k+4*N+2) * P_N^(k,0)(1-2r^2).

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

    P(:,n+2) = ((A2 + A3.*z).*P(:,n+1) - A4.*P(:,n))./A1;
end

N = 0:Nmax;
scale = sqrt(2*k + 4*N + 2);

RCL = P.*scale;

end
