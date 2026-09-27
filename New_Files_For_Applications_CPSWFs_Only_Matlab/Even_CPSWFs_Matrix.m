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
% Generate the symmetric tridiagonal matrix for EVEN CPSWFs.
%
% INPUTS
%   k : homogeneity degree
%   c : CPSWF bandwidth
%   m : matrix truncation size
%
% The mathematical entries are unchanged from the original code.
% This version fills the symmetric off-diagonal entries from one value
% to avoid any accidental mismatch.

function M = Even_CPSWFs_Matrix(k,c,m)

if ~(isscalar(k) && k >= 0 && k == floor(k))
    error('k must be a nonnegative integer.');
end

if ~(isscalar(c) && isreal(c) && c >= 0)
    error('c must be a nonnegative real scalar.');
end

if ~(isscalar(m) && m >= 1 && m == floor(m))
    error('m must be a positive integer.');
end

M = zeros(m,m);

M(1,1) = 4*pi^2*c^2*(k+1)/(k+2);

for i = 1:m-1

    off = -(4*pi^2*c^2*i*(k+i)) / ...
          ((k+2*i)*sqrt((k+2*i+1)*(k+2*i-1)));

    M(i,i+1) = off;
    M(i+1,i) = off;

    M(i+1,i+1) = ...
        4*i*(k+i+1) + ...
        4*pi^2*c^2 * ...
        ( ((k+i+1)^2)/((k+2*i+2)*(k+2*i+1)) + ...
          (i^2)/((k+2*i)*(k+2*i+1)) );
end

% Numerical safeguard: enforce exact symmetry.
M = 0.5*(M + M.');

end
