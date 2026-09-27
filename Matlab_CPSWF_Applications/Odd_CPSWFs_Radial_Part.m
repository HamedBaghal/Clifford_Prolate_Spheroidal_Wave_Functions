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
% Computes the polynomial RADIAL part of an ODD CPSWF.
%
% INPUTS
%   r : radius/radii in [0,1]
%   k : homogeneity degree
%   c : CPSWF bandwidth
%   m : odd Clifford-Legendre/Galerkin truncation size
%   n : CPSWF radial order, n = 1,...,m
%
% The odd angular factor x*Y_k supplies the r^(k+1) and angular dependence.
% Accordingly this routine returns the polynomial radial factor used in the
% original Odd_Eigenvalue_CPSWFs.m formula.
%
% STABILITY UPDATE
% ----------------
% Builds the complete odd radial Clifford-Legendre table by the stable
% Jacobi recurrence and performs one matrix-vector product.
%
% Relation:
%
%   Odd_CPSWFs_without_Yk(r,k,c,m,n)
%     = r^(k+1)/sqrt(2*pi)
%       * Odd_CPSWFs_Radial_Part(r,k,c,m,n).
%
% Function name and calling syntax are unchanged.

function G = Odd_CPSWFs_Radial_Part(r,k,c,m,n)

coeff = Odd_CPSWFs_Coefficient(k,c,m,n);

original_size = size(r);
rr = r(:);

RCL = odd_clp_radial_table(rr,m-1,k);

G = RCL * coeff;
G = reshape(G,original_size);

end


function RCL = odd_clp_radial_table(r,Nmax,k)
% Columns N=0,...,Nmax:
%
%   -sqrt(2*k+4*N+4) * P_N^(k+1,0)(1-2*r^2).

P = jacobi_alpha0_table(1 - 2*r.^2,Nmax,k+1);

N = 0:Nmax;
scale = -sqrt(2*k + 4*N + 4);

RCL = P.*scale;

end


function P = jacobi_alpha0_table(z,Nmax,alpha)

z = z(:);
beta = 0;

P = zeros(numel(z),Nmax+1);
P(:,1) = 1;

if Nmax == 0
    return
end

P(:,2) = 0.5*((alpha-beta) + ...
              (alpha+beta+2).*z);

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
