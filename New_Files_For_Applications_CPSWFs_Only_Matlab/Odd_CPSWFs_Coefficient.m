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
% Generates the coefficients of the n-th ODD CPSWF expanded in the
% normalized odd Clifford-Legendre basis.
%
% INPUTS
%   k : homogeneity degree
%   c : CPSWF bandwidth
%   m : matrix truncation size
%   n : radial CPSWF order, MATLAB indexing n = 1,...,m
%
% CORRECTION
% ----------
% Your original code sorted diag(D) but did NOT apply the same permutation
% to V before selecting V(:,n).  That is the same eigenvector-ordering
% problem we already corrected in the even CPSWF code.
%
% This version sorts eigenvalues and eigenvectors together and fixes the
% arbitrary eigenvector sign for reproducible output.
%
% Function name and calling syntax are unchanged.

function W = Odd_CPSWFs_Coefficient(k,c,m,n)

validateattributes(n,{'numeric'},{'scalar','integer','>=',1,'<=',m});

A = Odd_CPSWFs_Matrix(k,c,m);
A = 0.5*(A + A.');

[V,D] = eig(A);

[~,idx] = sort(real(diag(D)),'ascend');
V = real(V(:,idx));

W = V(:,n);

% Reproducible sign convention.
[~,imax] = max(abs(W));

if W(imax) < 0
    W = -W;
end

end
