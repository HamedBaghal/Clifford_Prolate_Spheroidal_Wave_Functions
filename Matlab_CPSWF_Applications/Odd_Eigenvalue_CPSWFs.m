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
% Computes the ODD CPSWF finite-Fourier eigenvalue.
%
% The mathematical formula is kept unchanged from your original code:
%
%   q = N(1)*sqrt(2*k+4)*pi^(k+2)*c^(k+1)*i^(k+1)
%       /(Gamma(k+3)*R_odd(0)).
%
% IMPROVEMENTS
% ------------
%   - corrected/stable odd CPSWF coefficients;
%   - stable Odd_CPSWFs_Radial_Part;
%   - logarithmic evaluation of the large real factor
%       pi^(k+2)*c^(k+1)/Gamma(k+3).
%
% Function name and calling syntax are unchanged.

function q = Odd_Eigenvalue_CPSWFs(k,c,m,n)

validateattributes(k,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(c,{'numeric'},{'scalar','real','positive'});
validateattributes(m,{'numeric'},{'scalar','integer','positive'});
validateattributes(n,{'numeric'},{'scalar','integer','>=',1,'<=',m});

N = Odd_CPSWFs_Coefficient(k,c,m,n);

radial0 = Odd_CPSWFs_Radial_Part(0,k,c,m,n);

if ~isfinite(radial0) || radial0 == 0
    error('Odd_CPSWFs_Radial_Part(0,...) is zero or non-finite.');
end

logScale = (k+2)*log(pi) + ...
           (k+1)*log(c) - ...
           gammaln(k+3);

scale = sqrt(2*k+4) * exp(logScale) * (1i)^(k+1);

q = (N(1)/radial0) * scale;

end
