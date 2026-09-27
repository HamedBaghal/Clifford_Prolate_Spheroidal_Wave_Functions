% References
%
% 1. Ghaffari, H. B., Hogan, J. A., & Lakey, J. D. (2022).
%    Properties of Clifford-Legendre Polynomials.
%    Advances in Applied Clifford Algebras, 32(1), 1-25.
%    https://doi.org/10.1007/s00006-021-01179-8
%
% 2. H. Baghal Ghaffari, "Higher-dimensional prolate spheroidal wave
%    functions," Ph.D. dissertation, The University of Newcastle, 2022.
clear;
close all;
clc;

%% 1. PARAMETERS

L = 128;
Npix = 2*L + 1;

c_CPSWF = L/2;
c_PSWF = 2*pi*c_CPSWF;

% Equal-budget target taken from the full CPSWF / independent-PSWF run:
%
%   46,458 real scalar basis coefficients per channel
%   4 x 46,458 = 185,832 real coefficients in total.
%
% The Fourier--Bessel cutoff is chosen automatically so that the number
% of retained REAL Fourier--Bessel basis functions per channel is as
% close as possible to this target.
target_real_basis_per_channel = 46458;
target_total_real_coefficients = 4*target_real_basis_per_channel;

% Search sufficiently beyond the previous common-cutoff value c_PSWF.
% The expected equal-budget cutoff is around 432.
lambda_search_max = 1.20*c_PSWF;

Nr = 320;
Ntheta = 2048;

% Radial grid used only for efficient synthesis before interpolation to
% Cartesian disk pixels.  This does not affect coefficient integration.
NrSynth = 2049;

rgb_name = '0000_rgb.tiff';
nir_name = '0000_nir.tiff';

pad_value = 0;

%% 2. CHECK INPUTS

addpath(pwd);

if isempty(which('lgwt'))
    error('lgwt.m was not found on the MATLAB path.');
end

if ~isfile(rgb_name)
    error('RGB image %s was not found.',rgb_name);
end

if ~isfile(nir_name)
    error('NIR image %s was not found.',nir_name);
end

fprintf('\n============================================================\n');
fprintf('FOURIER--BESSEL RGB-NIR BASELINE\n');
fprintf('============================================================\n');
fprintf('RGB image                    : %s\n',rgb_name);
fprintf('NIR image                    : %s\n',nir_name);
fprintf('Grid                         : %d x %d\n',Npix,Npix);
fprintf('Reference PSWF c             : %.12f\n',c_PSWF);
fprintf('Target real basis/channel    : %d\n',target_real_basis_per_channel);
fprintf('Target total real coeffs     : %d\n',target_total_real_coefficients);
fprintf('Cutoff search upper bound    : %.12f\n',lambda_search_max);
fprintf('Radial GL nodes              : %d\n',Nr);
fprintf('Angular nodes                : %d\n',Ntheta);
fprintf('Synthesis radial grid        : %d\n',NrSynth);
fprintf('Basis                        : real orthonormal Fourier--Bessel\n');
fprintf('============================================================\n\n');

%% 3. READ REGISTERED RGB AND NIR IMAGES

rawRGB = imread(rgb_name);
rawNIR = imread(nir_name);

if ndims(rawRGB) ~= 3 || size(rawRGB,3) < 3
    error('RGB input must contain at least three channels.');
end

if ndims(rawNIR) == 3
    rawNIR = rawNIR(:,:,1);
end

if size(rawRGB,1) ~= size(rawNIR,1) || ...
   size(rawRGB,2) ~= size(rawNIR,2)
    error('RGB and NIR images must be spatially registered and the same size.');
end

RGB0 = local_to_unit_double(rawRGB(:,:,1:3));
NIR0 = local_to_unit_double(rawNIR);

origH = size(RGB0,1);
origW = size(RGB0,2);

%% 4. SAME CLEAN CENTERED CIRCULAR CROP AS CPSWF/PSWF TESTS

cx = (origW + 1)/2;
cy = (origH + 1)/2;
circle_radius_px = (min(origH,origW) - 1)/2;

x_left   = cx - circle_radius_px;
x_right  = cx + circle_radius_px;
y_top    = cy - circle_radius_px;
y_bottom = cy + circle_radius_px;

[xsrc,ysrc] = meshgrid(1:origW,1:origH);
[xfit,yfit] = meshgrid(linspace(x_left,x_right,Npix), ...
                       linspace(y_top,y_bottom,Npix));

RGB = zeros(Npix,Npix,3);
for ch = 1:3
    RGB(:,:,ch) = interp2(xsrc,ysrc,RGB0(:,:,ch), ...
                          xfit,yfit,'linear',pad_value);
end

NIR = interp2(xsrc,ysrc,NIR0,xfit,yfit,'linear',pad_value);

RGB = min(max(RGB,0),1);
NIR = min(max(NIR,0),1);

[xcart,ycart] = meshgrid(linspace(-1,1,Npix), ...
                         linspace(-1,1,Npix));
unit_disk = (xcart.^2 + ycart.^2) <= 1;

for ch = 1:3
    tmp = RGB(:,:,ch);
    tmp(~unit_disk) = pad_value;
    RGB(:,:,ch) = tmp;
end
NIR(~unit_disk) = pad_value;

fprintf('Original registered size     : %d x %d\n',origH,origW);
fprintf('Circle radius                : %.4f px\n\n',circle_radius_px);

%% 5. CARTESIAN DISK SAMPLES

[X,Y] = meshgrid(-L:L,-L:L);
inside = sqrt(X.^2 + Y.^2) <= L;

x_disk = X(inside);
y_disk = Y(inside);

r_pix = sqrt(x_disk.^2 + y_disk.^2)/L;
theta_pix = atan2(y_disk,x_disk);

Rimg = RGB(:,:,1);
Gimg = RGB(:,:,2);
Bimg = RGB(:,:,3);

Ftrue = [ ...
    NIR(inside), ...
    Rimg(inside), ...
    Gimg(inside), ...
    Bimg(inside)];

Ndisk = nnz(inside);

fprintf('Pixels inside disk           : %d\n',Ndisk);

%% 6. POLAR GAUSS--LEGENDRE QUADRATURE

[rq,wr] = lgwt(Nr,0,1);
rq = rq(:);
wr = wr(:);

thetaq = (2*pi/Ntheta)*(0:Ntheta-1);
dtheta = 2*pi/Ntheta;

[ThetaQ,RQ] = meshgrid(thetaq,rq);

XQ = RQ.*cos(ThetaQ);
YQ = RQ.*sin(ThetaQ);

% Interpolate each cropped channel to polar quadrature nodes.
FQ = zeros(Nr,Ntheta,4);

FQ(:,:,1) = interp2(xcart,ycart,NIR,XQ,YQ,'linear',0);
FQ(:,:,2) = interp2(xcart,ycart,Rimg,XQ,YQ,'linear',0);
FQ(:,:,3) = interp2(xcart,ycart,Gimg,XQ,YQ,'linear',0);
FQ(:,:,4) = interp2(xcart,ycart,Bimg,XQ,YQ,'linear',0);

radial_weight = wr.*rq;

%% 7. CHOOSE FOURIER--BESSEL CUTOFF BY EQUAL REAL-COEFFICIENT BUDGET

% For n=0 each Bessel zero contributes one real basis function.
% For n>0 each zero contributes the cosine/sine pair, hence two real
% basis functions.  We first enumerate all roots below lambda_search_max,
% sort them by radial frequency j_{n,q}, and choose the cutoff whose
% cumulative REAL basis count is closest to the CPSWF/PSWF target.

nmax_search = floor(lambda_search_max);
roots_search = cell(nmax_search+1,1);

all_roots = [];
all_mult  = [];

fprintf('Finding candidate Bessel zeros for equal-budget matching ...\n');
tic

for n = 0:nmax_search

    z = local_bessel_zeros_below(n,lambda_search_max);
    roots_search{n+1} = z(:);

    if isempty(z)
        continue
    end

    all_roots = [all_roots; z(:)]; %#ok<AGROW>

    if n == 0
        all_mult = [all_mult; ones(numel(z),1)]; %#ok<AGROW>
    else
        all_mult = [all_mult; 2*ones(numel(z),1)]; %#ok<AGROW>
    end
end

[sorted_roots,ord] = sort(all_roots);
sorted_mult = all_mult(ord);
cum_real_basis = cumsum(sorted_mult);

[~,ibest] = min(abs(cum_real_basis-target_real_basis_per_channel));

% Put the cutoff just above the selected zero so that the selected shell
% is included robustly despite floating-point comparisons.
lambda_max = sorted_roots(ibest) + ...
             100*eps(max(1,sorted_roots(ibest)));

matched_real_basis_per_channel = cum_real_basis(ibest);
matched_total_real_coefficients = 4*matched_real_basis_per_channel;

% Keep only roots below the automatically selected cutoff.
roots_by_n = cell(nmax_search+1,1);
last_n = -1;
real_basis_count = 0;

for n = 0:nmax_search

    z = roots_search{n+1};
    z = z(z <= lambda_max);

    if isempty(z)
        continue
    end

    roots_by_n{n+1} = z;
    last_n = n;

    if n == 0
        real_basis_count = real_basis_count + numel(z);
    else
        real_basis_count = real_basis_count + 2*numel(z);
    end
end

zero_time = toc;

fprintf('\nEqual-budget cutoff selected:\n');
fprintf('  lambda_max                 : %.12f\n',lambda_max);
fprintf('  target basis/channel        : %d\n',target_real_basis_per_channel);
fprintf('  achieved basis/channel      : %d\n',real_basis_count);
fprintf('  basis-count difference      : %+d\n', ...
        real_basis_count-target_real_basis_per_channel);
fprintf('  target total coefficients   : %d\n',target_total_real_coefficients);
fprintf('  achieved total coefficients : %d\n',4*real_basis_count);
fprintf('  largest angular order n     : %d\n',last_n);
fprintf('  Bessel-zero setup time      : %.3f s\n\n',zero_time);

%% 8. PROJECT AND RECONSTRUCT

Frec = zeros(Ndisk,4);

% Uniform radial synthesis grid.
rs = linspace(0,1,NrSynth).';

fprintf('Computing Fourier--Bessel coefficients and reconstruction ...\n');
tic

for n = 0:last_n

    roots = roots_by_n{n+1};
    if isempty(roots)
        continue
    end

    nq = numel(roots);

    % Radial basis at coefficient quadrature radii.
    Jq = besselj(n,rq*roots.');

    if n == 0
        normc = 1 ./ (sqrt(pi)*abs(besselj(1,roots)));
    else
        normc = sqrt(2) ./ ...
            (sqrt(pi)*abs(besselj(n+1,roots)));
    end

    Bq = bsxfun(@times,Jq,normc.');

    cn = cos(n*thetaq).';
    sn = sin(n*thetaq).';

    % Angular integrals for all four channels.
    Ic = zeros(Nr,4);
    Is = zeros(Nr,4);

    for ch = 1:4
        A = FQ(:,:,ch);
        Ic(:,ch) = dtheta*(A*cn);

        if n > 0
            Is(:,ch) = dtheta*(A*sn);
        end
    end

    % Orthonormal Fourier--Bessel coefficients.
    %
    % coeff = int_0^1 [angular integral] radial_basis(r) r dr.
    Ac = Bq.' * bsxfun(@times,radial_weight,Ic);

    if n > 0
        As = Bq.' * bsxfun(@times,radial_weight,Is);
    end

    % Synthesis on a fine radial grid, then interpolate each radial profile
    % to the Cartesian disk radii.
    Js = besselj(n,rs*roots.');
    Bs = bsxfun(@times,Js,normc.');

    Pc = Bs*Ac;  % NrSynth x 4

    if n > 0
        Ps = Bs*As;
    end

    ctp = cos(n*theta_pix);
    stp = sin(n*theta_pix);

    for ch = 1:4
        vc = interp1(rs,Pc(:,ch),r_pix,'pchip');

        if n == 0
            Frec(:,ch) = Frec(:,ch) + vc;
        else
            vs = interp1(rs,Ps(:,ch),r_pix,'pchip');
            Frec(:,ch) = Frec(:,ch) + vc.*ctp + vs.*stp;
        end
    end

    if mod(n,20)==0 || n==last_n
        fprintf('  n=%3d, radial roots=%3d\n',n,nq);
    end
end

runtime = toc;

%% 9. ERRORS

Ntrue = Ftrue(:,1);
Rtrue = Ftrue(:,2);
Gtrue = Ftrue(:,3);
Btrue = Ftrue(:,4);

Nrec = Frec(:,1);
Rrec = Frec(:,2);
Grec = Frec(:,3);
Brec = Frec(:,4);

combined_rel = norm(Frec(:)-Ftrue(:))/norm(Ftrue(:));

nir_rel = norm(Nrec-Ntrue)/norm(Ntrue);

rgb_true_vec = [Rtrue;Gtrue;Btrue];
rgb_rec_vec  = [Rrec;Grec;Brec];

rgb_rel = norm(rgb_rec_vec-rgb_true_vec)/norm(rgb_true_vec);

R_rel = norm(Rrec-Rtrue)/norm(Rtrue);
G_rel = norm(Grec-Gtrue)/norm(Gtrue);
B_rel = norm(Brec-Btrue)/norm(Btrue);

mse4 = mean((Frec(:)-Ftrue(:)).^2);
psnr4 = 10*log10(1/max(mse4,realmin));

mseRGB = mean((rgb_rec_vec-rgb_true_vec).^2);
psnrRGB = 10*log10(1/max(mseRGB,realmin));

fprintf('\n============================================================\n');
fprintf('EQUAL-BUDGET FOURIER--BESSEL RESULTS\n');
fprintf('============================================================\n');
fprintf('Selected lambda_max            : %.12f\n',lambda_max);
fprintf('Largest angular order n       : %d\n',last_n);
fprintf('Target basis funcs/channel    : %d\n',target_real_basis_per_channel);
fprintf('Real basis funcs/channel      : %d\n',real_basis_count);
fprintf('Target total real coeffs      : %d\n',target_total_real_coefficients);
fprintf('Total real coefficients       : %d\n',4*real_basis_count);
fprintf('Processing time               : %.3f s\n',runtime);
fprintf('\n');
fprintf('Combined 4-channel rel. error : %.12e\n',combined_rel);
fprintf('NIR relative error            : %.12e\n',nir_rel);
fprintf('RGB relative error            : %.12e\n',rgb_rel);
fprintf('R relative error              : %.12e\n',R_rel);
fprintf('G relative error              : %.12e\n',G_rel);
fprintf('B relative error              : %.12e\n',B_rel);
fprintf('4-channel PSNR                : %.6f dB\n',psnr4);
fprintf('RGB PSNR                      : %.6f dB\n',psnrRGB);
fprintf('============================================================\n');

%% 10. RECONSTRUCTED IMAGES

NIR_rec_img = zeros(Npix,Npix);
R_rec_img   = zeros(Npix,Npix);
G_rec_img   = zeros(Npix,Npix);
B_rec_img   = zeros(Npix,Npix);

NIR_rec_img(inside)=Nrec;
R_rec_img(inside)=Rrec;
G_rec_img(inside)=Grec;
B_rec_img(inside)=Brec;

RGB_rec = cat(3,R_rec_img,G_rec_img,B_rec_img);
RGB_rec = min(max(RGB_rec,0),1);
NIR_rec_img = min(max(NIR_rec_img,0),1);

figure('Name','Equal-budget Fourier--Bessel RGB-NIR reconstruction');

subplot(2,2,1)
imshow(RGB)
title('Original RGB')

subplot(2,2,2)
imshow(RGB_rec)
title(sprintf('Fourier--Bessel RGB, rel.err %.3e',rgb_rel))

subplot(2,2,3)
imshow(NIR,[])
title('Original NIR')

subplot(2,2,4)
imshow(NIR_rec_img,[])
title(sprintf('Fourier--Bessel NIR, rel.err %.3e',nir_rel))

%% 11. SAVE

imwrite(RGB_rec,'0000_RGBNIR_FourierBessel_EqualBudget_RGB_L128.png');
imwrite(NIR_rec_img,'0000_RGBNIR_FourierBessel_EqualBudget_NIR_L128.png');

save('0000_RGBNIR_FourierBessel_EqualBudget_L128_results.mat', ...
    'L','lambda_max','lambda_search_max','Nr','Ntheta','NrSynth', ...
    'target_real_basis_per_channel','target_total_real_coefficients', ...
    'last_n','real_basis_count','runtime', ...
    'combined_rel','nir_rel','rgb_rel', ...
    'R_rel','G_rel','B_rel','psnr4','psnrRGB');

fprintf('\nSaved:\n');
fprintf('  0000_RGBNIR_FourierBessel_EqualBudget_RGB_L128.png\n');
fprintf('  0000_RGBNIR_FourierBessel_EqualBudget_NIR_L128.png\n');
fprintf('  0000_RGBNIR_FourierBessel_EqualBudget_L128_results.mat\n');

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function A = local_to_unit_double(A)

    if isinteger(A)
        A = double(A)/double(intmax(class(A)));
    else
        A = double(A);
        if max(A(:)) > 1
            A = A/max(A(:));
        end
    end
end

function roots = local_bessel_zeros_below(n,xmax)
%
% Robustly locate positive zeros of J_n(x) below xmax by sign-change
% scanning followed by fzero refinement.
%
% This is a PRECOMPUTATION routine for the accuracy baseline, not the
% asymptotically fast zero machinery of an FDHT implementation.

    if xmax <= 0
        roots = [];
        return
    end

    % J_n has its first positive zero strictly larger than n for n>=1.
    if n == 0
        xmin = 1e-8;
    else
        xmin = max(1e-6,0.90*n);
    end

    if xmin >= xmax
        roots = [];
        return
    end

    dx = pi/10;
    x = xmin:dx:(xmax+dx);

    if x(end) < xmax
        x(end+1)=xmax;
    end

    y = besselj(n,x);

    % Ignore underflow-level values before the oscillatory region.
    tiny = abs(y) < 1e-280;
    y(tiny) = NaN;

    roots = [];

    for j = 1:numel(x)-1

        y1 = y(j);
        y2 = y(j+1);

        if isnan(y1) || isnan(y2)
            continue
        end

        if y1 == 0
            rr = x(j);
        elseif y1*y2 < 0
            rr = fzero(@(t)besselj(n,t),[x(j),x(j+1)]);
        else
            continue
        end

        if rr > 0 && rr <= xmax*(1+1e-12)
            if isempty(roots) || abs(rr-roots(end)) > 1e-7
                roots(end+1,1)=rr; %#ok<AGROW>
            end
        end
    end

    roots = roots(roots <= xmax + 1e-10);
end
