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
% Generate the coefficients of the n-th EVEN CPSWF expanded in the
% normalized even Clifford-Legendre basis.
%
% INPUTS
%   k : homogeneity degree
%   c : CPSWF bandwidth
%   m : matrix truncation size
%   n : CPSWF radial order, using MATLAB indexing n = 1,...,m
%
% NUMERICAL/CORRECTNESS UPDATE
% The original code sorted the eigenvalues but did not apply the same
% permutation to the eigenvectors.  Here eigenvalues and eigenvectors are
% sorted together.  The arbitrary eigenvector sign is also fixed for
% reproducible output.
%
% The function name and calling syntax are unchanged.

function W = Even_CPSWFs_Coefficient(k,c,m,n)

if ~(isscalar(n) && n >= 1 && n <= m && n == floor(n))
    error('n must be an integer satisfying 1 <= n <= m.');
end

A = Even_CPSWFs_Matrix(k,c,m);
A = 0.5*(A + A.');

[V,D] = eig(A);

% For c=0 the differential eigenvalues increase with radial order,
% so the ascending ordering is the natural continuation for c>0.
[~,idx] = sort(real(diag(D)),'ascend');
V = V(:,idx);

W = real(V(:,n));

% Eigenvectors are defined only up to sign.  Fix the sign by forcing
% the largest-magnitude coefficient to be positive.
[~,imax] = max(abs(W));

if W(imax) < 0
    W = -W;
end

end
