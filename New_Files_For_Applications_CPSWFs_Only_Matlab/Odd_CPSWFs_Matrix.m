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
% Generates the symmetric tridiagonal matrix for ODD CPSWFs.
%
% INPUTS
%   k : homogeneity degree
%   c : CPSWF bandwidth
%   m : matrix truncation size
%
% The mathematical entries are unchanged from your original code.
%
% Useful identity
% ---------------
% With the conventions used in these files,
%
%   Odd_CPSWFs_Matrix(k,c,m)
%     = Even_CPSWFs_Matrix(k+1,c,m) + 4*(k+1)*I.
%
% Therefore the odd k-family has the SAME eigenvectors as the even
% (k+1)-family; only the differential eigenvalues are shifted by a
% constant.  This is consistent with the odd factor x*Y_k and angular
% order k+1.
%
% Function name and calling syntax are unchanged.

function M = Odd_CPSWFs_Matrix(k,c,m)

validateattributes(k,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(c,{'numeric'},{'scalar','real','nonnegative'});
validateattributes(m,{'numeric'},{'scalar','integer','positive'});

M = zeros(m,m);

% Diagonal entries: unchanged from the original implementation.
for i = 1:m

    M(i,i) = ...
        4*i*(k+i) + ...
        4*pi^2*c^2 * ...
        ( ((i+k)^2)/((2*i+k-1)*(2*i+k)) + ...
          (i^2)/((2*i+k+1)*(2*i+k)) );
end

% Symmetric off-diagonal entries.
for i = 1:m-1

    off = -(4*pi^2*c^2*i*(k+i+1)) / ...
          ((k+2*i+1)*sqrt((k+2*i+2)*(k+2*i)));

    M(i,i+1) = off;
    M(i+1,i) = off;
end

% Numerical safeguard.
M = 0.5*(M + M.');

end
