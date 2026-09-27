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
% Computes an ODD CPSWF with the angular Clifford factor omitted.
%
% The user's 2-D odd angular convention is
%
%      x*Y_k(x)
%
% or, on the unit circle, the two components
%
%      -cos((k+1)*theta),  -sin((k+1)*theta).
%
% Those angular factors are intentionally NOT included here.  For example,
% in a 2-D construction one may use
%
%   q0 = -cos((k+1)*theta) .* Odd_CPSWFs_without_Yk(r,k,c,m,n);
%   q3 = -sin((k+1)*theta) .* Odd_CPSWFs_without_Yk(r,k,c,m,n);
%
% This function preserves the sign convention of the original odd
% Clifford-Legendre implementation.
%
% NUMERICAL UPDATE
% ----------------
% Uses:
%   - corrected eigenvector ordering;
%   - stable Jacobi evaluation;
%   - one matrix-vector multiplication instead of repeated unstable sums.
%
% Function name and calling syntax are unchanged.

function G = Odd_CPSWFs_without_Yk(r,k,c,m,n)

coeff = Odd_CPSWFs_Coefficient(k,c,m,n);

original_size = size(r);
rr = r(:);

CL = odd_clp_table(rr,m-1,k);

G = CL * coeff;
G = reshape(G,original_size);

end


function CL = odd_clp_table(r,Nmax,k)
% Columns N=0,...,Nmax are
%
%   -sqrt(2*k+4*N+4)/sqrt(2*pi)
%    * r^(k+1) * P_N^(k+1,0)(1-2*r^2).

P = jacobi_alpha0_table(1 - 2*r.^2,Nmax,k+1);

N = 0:Nmax;
scale = -sqrt(2*k + 4*N + 4)/sqrt(2*pi);

CL = (r.^(k+1)).*P;
CL = CL.*scale;

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
