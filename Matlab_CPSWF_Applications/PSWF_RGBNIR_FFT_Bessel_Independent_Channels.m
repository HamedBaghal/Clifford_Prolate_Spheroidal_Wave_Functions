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
Mgrid = Npix;

c_CPSWF = L/2;                  % 64
c_PSWF  = 2*pi*c_CPSWF;         % pi*L
beta    = c_PSWF/(pi*L);        % 1

T = 1e-5;
m = 220;

% Change these two names for another registered pair.
rgb_name = '0000_rgb.tiff';
nir_name = '0000_nir.tiff';

pad_value = 0;

cache_file = ...
    'CPSWF_FFTBessel_Precompute_L128_c64_m220_T1e-5.mat';

% Automatically derive image prefix.
[~,rgb_stem,~] = fileparts(rgb_name);
prefix = regexprep(rgb_stem,'_rgb$','');

rgb_output = sprintf( ...
    '%s_RGBNIR_PSWF_FFTBessel_RGB_L128.png',prefix);

nir_output = sprintf( ...
    '%s_RGBNIR_PSWF_FFTBessel_NIR_L128.png',prefix);

mat_output = sprintf( ...
    '%s_RGBNIR_PSWF_FFTBessel_L128_results.mat',prefix);

% Same-image CPSWF result candidates.
cpswf_reference_candidates = { ...
    sprintf('%s_RGBNIR_CPSWF_FFTBessel_OPT_L128_results.mat',prefix), ...
    sprintf('%s_RGBNIR_CPSWF_FFTBessel_L128_results.mat',prefix)};

alphaM = 2*pi*L/Mgrid;

%% 2. CHECK INPUTS

if ~isfile(rgb_name)
    error('RGB image %s was not found.',rgb_name);
end

if ~isfile(nir_name)
    error('NIR image %s was not found.',nir_name);
end

if ~isfile(cache_file)
    error(['Reusable cache %s was not found. ', ...
           'Run the optimized CPSWF code first.'],cache_file);
end

fprintf('\n============================================================\n');
fprintf('FOUR INDEPENDENT PSWF CHANNELS — OPTIMIZED FFT--BESSEL\n');
fprintf('============================================================\n');
fprintf('RGB image                   : %s\n',rgb_name);
fprintf('NIR image                   : %s\n',nir_name);
fprintf('Grid                        : %d x %d\n',Npix,Npix);
fprintf('L                           : %d\n',L);
fprintf('CPSWF bandwidth c           : %.12g\n',c_CPSWF);
fprintf('Equivalent scalar PSWF c    : %.12f\n',c_PSWF);
fprintf('beta                        : %.12g\n',beta);
fprintf('Omega_T threshold           : %.3e\n',T);
fprintf('Galerkin dimension m        : %d\n',m);
fprintf('Cache                       : %s\n',cache_file);
fprintf('============================================================\n\n');

%% 3. READ AND CROP REGISTERED RGB/NIR IMAGES

rawRGB = imread(rgb_name);
rawNIR = imread(nir_name);

if ndims(rawRGB)~=3 || size(rawRGB,3)<3
    error('RGB input must contain at least three channels.');
end

if ndims(rawNIR)==3
    rawNIR = rawNIR(:,:,1);
end

if size(rawRGB,1)~=size(rawNIR,1) || ...
   size(rawRGB,2)~=size(rawNIR,2)
    error('RGB and NIR images must have identical registered dimensions.');
end

RGB0 = local_to_unit_double(rawRGB(:,:,1:3));
NIR0 = local_to_unit_double(rawNIR);

origH = size(RGB0,1);
origW = size(RGB0,2);

cx = (origW+1)/2;
cy = (origH+1)/2;

circle_radius_px = (min(origH,origW)-1)/2;

x_left   = cx-circle_radius_px;
x_right  = cx+circle_radius_px;
y_top    = cy-circle_radius_px;
y_bottom = cy+circle_radius_px;

[xsrc,ysrc] = meshgrid(1:origW,1:origH);

[xfit,yfit] = meshgrid( ...
    linspace(x_left,x_right,Npix), ...
    linspace(y_top,y_bottom,Npix));

RGB = zeros(Npix,Npix,3);

for ch=1:3
    RGB(:,:,ch) = interp2( ...
        xsrc,ysrc,RGB0(:,:,ch), ...
        xfit,yfit,'linear',pad_value);
end

NIR = interp2( ...
    xsrc,ysrc,NIR0, ...
    xfit,yfit,'linear',pad_value);

RGB = min(max(RGB,0),1);
NIR = min(max(NIR,0),1);

[Xgrid,Ygrid] = meshgrid(-L:L,-L:L);

inside = (Xgrid.^2 + Ygrid.^2) <= L^2;

for ch=1:3
    tmp = RGB(:,:,ch);
    tmp(~inside)=pad_value;
    RGB(:,:,ch)=tmp;
end

NIR(~inside)=pad_value;

x_disk = Xgrid(inside);
y_disk = Ygrid(inside);

r_pix = sqrt(x_disk.^2+y_disk.^2)/L;
theta_pix = atan2(y_disk,x_disk);

Ndisk = nnz(inside);

Nval = NIR(inside);

Rimg = RGB(:,:,1);
Gimg = RGB(:,:,2);
Bimg = RGB(:,:,3);

Rval = Rimg(inside);
Gval = Gimg(inside);
Bval = Bimg(inside);

fprintf('Original registered size    : %d x %d\n',origH,origW);
fprintf('Pixels inside disk          : %d\n\n',Ndisk);

%% 4. UNIQUE SPATIAL RADII

pixRadiusSq = x_disk.^2 + y_disk.^2;

[uniquePixRadiusSq,~,pixRadiusGroup] = unique(pixRadiusSq);

rUniquePix = sqrt(uniquePixRadiusSq)/L;

NuniquePix = numel(rUniquePix);

fprintf('Unique spatial radii        : %d\n',NuniquePix);

%% 5. EXACT CENTERED 2-D DFT

ivec = -L:L;

E = exp(-2*pi*1i*(ivec(:)*ivec(:).')/Mgrid);

fprintf('Computing exact centered DFT of four channels ...\n');

tt=tic;

F0hat  = E*NIR       *E.';       % NIR
F1hat  = E*RGB(:,:,1)*E.';       % R
F2hat  = E*RGB(:,:,2)*E.';       % G
F3hat  = E*RGB(:,:,3)*E.';       % B

dft_time=toc(tt);

% Check exact discrete inversion.
NIR_back = real(E'*F0hat*conj(E))/Mgrid^2;

dft_inverse_error = ...
    norm(NIR_back(:)-NIR(:))/max(norm(NIR(:)),eps);

fprintf('DFT time                    : %.3f s\n',dft_time);
fprintf('Inverse DFT check           : %.3e\n\n',dft_inverse_error);

%% 6. FREQUENCY GEOMETRY / UNIQUE RADII

[J1,J2] = meshgrid(ivec,ivec);

Jabs = sqrt(J1.^2 + J2.^2);
PhiJ = atan2(J2,J1);

zeroMask = (Jabs==0);
nonzeroMask = ~zeroMask;

PhiNZ = PhiJ(nonzeroMask);

J1nz = J1(nonzeroMask);
J2nz = J2(nonzeroMask);

% Columns = [NIR,R,G,B].
Ffreq = [ ...
    F0hat(nonzeroMask), ...
    F1hat(nonzeroMask), ...
    F2hat(nonzeroMask), ...
    F3hat(nonzeroMask)];

Fzero = [ ...
    F0hat(zeroMask), ...
    F1hat(zeroMask), ...
    F2hat(zeroMask), ...
    F3hat(zeroMask)];

sqNZ = J1nz.^2 + J2nz.^2;

[uniqueSq,~,freqRadiusGroup] = unique(sqNZ);

rhoUnique = alphaM*sqrt(uniqueSq);

NuniqueFreq = numel(rhoUnique);
Nfreq = numel(freqRadiusGroup);

RadiusSum = sparse( ...
    freqRadiusGroup, ...
    (1:Nfreq).', ...
    ones(Nfreq,1), ...
    NuniqueFreq,Nfreq);

fprintf('Nonzero frequency points    : %d\n',Nfreq);
fprintf('Unique frequency radii      : %d\n\n',NuniqueFreq);

%% 7. LOAD SAME CPSWF/PSWF PRECOMPUTATION CACHE

fprintf('Loading reusable radial/eigensystem cache ...\n');

tt=tic;
C = load(cache_file);
cache_load_time=toc(tt);

required_fields = { ...
    'cache_L','cache_c','cache_m','cache_T', ...
    'rhoUniqueCache','nrByEll','lastEll','VretCell','JoverRho'};

for q=1:numel(required_fields)
    if ~isfield(C,required_fields{q})
        error('Cache is missing required field: %s',required_fields{q});
    end
end

cache_ok = ...
    C.cache_L==L && ...
    abs(C.cache_c-c_CPSWF)<1e-14 && ...
    C.cache_m==m && ...
    abs(C.cache_T-T)<1e-14 && ...
    numel(C.rhoUniqueCache)==numel(rhoUnique) && ...
    max(abs(C.rhoUniqueCache(:)-rhoUnique(:)))<1e-12;

if ~cache_ok
    error('Existing cache does not match this L,c,m,T/frequency grid.');
end

VretCell = C.VretCell;
nrByEll  = C.nrByEll;
lastEll  = C.lastEll;
JoverRho = C.JoverRho;

fprintf('Cache loaded                : %.3f s\n',cache_load_time);
fprintf('Largest retained ell        : %d\n\n',lastEll);

%% 8. INITIALIZE FOUR INDEPENDENT SCALAR RECONSTRUCTIONS

% Columns = [NIR,R,G,B].
Rec = complex(zeros(Ndisk,4));

% Diagnostic coefficient magnitudes per retained radial mode.
CoeffMagN = [];
CoeffMagR = [];
CoeffMagG = [];
CoeffMagB = [];

ModeEll = [];
ModeN = [];

% Number of scalar basis functions:
%
% ell=0 : one cosine mode per radial mode
% ell>0 : cosine + sine = two real modes per radial mode.
stored_nonnegative_modes = 0;
real_basis_per_channel = 0;

phaseFreq = exp(1i*PhiNZ);
harmFreq = ones(Nfreq,1);

phasePix = exp(1i*theta_pix);
harmPix = ones(Ndisk,1);

rPowUnique = ones(NuniquePix,1);

time_angular = 0;
time_coeff = 0;
time_spatial = 0;

max_coeff_imag_ratio = 0;

fprintf('Starting independent scalar PSWF reconstruction ...\n');

main_timer=tic;

%% 9. MAIN ANGULAR LOOP

for ell=0:lastEll

    nr = nrByEll(ell+1);
    Vret = VretCell{ell+1};

    fprintf('  ell=%3d : %3d radial modes',ell,nr);

    if ell==0
        fprintf(' (cosine only)\n');
    else
        fprintf(' (cosine + sine)\n');
    end

    %% 9A. ANGULAR FOURIER SUMS FOR ALL FOUR CHANNELS

    tt=tic;

    Cphi = real(harmFreq);
    Sphi = imag(harmFreq);

    % Each result is NuniqueFreq x 4.
    Csum = RadiusSum*(Ffreq.*Cphi);

    if ell>=1
        Ssum = RadiusSum*(Ffreq.*Sphi);
    else
        Ssum = zeros(NuniqueFreq,4);
    end

    time_angular = time_angular + toc(tt);

    %% 9B. CLOSED FFT--BESSEL RADIAL PROJECTION

    tt=tic;

    s = 0:m-1;

    orders = ell+2*s+1;

    scale = sqrt(2*ell+4*s+2)/sqrt(2*pi);

    % Bbasis(q,s) =
    %   kappa_{s,ell} J_{ell+2s+1}(rho_q)/rho_q.
    Bbasis = JoverRho(:,orders).*scale;

    pref = 2*pi*(1i^ell)/Mgrid^2;

    % Scalar cosine coefficients, nr x 4:
    %
    % [N_c, R_c, G_c, B_c].
    basisCos = Bbasis.'*Csum;

    coeffCos = pref*(Vret.'*basisCos);

    % Zero frequency contributes only to ell=0 cosine coefficients.
    if ell==0

        zeroKernel = ...
            0.5*sqrt(2/(2*pi))*Vret(1,:).';

        coeffCos = ...
            coeffCos + pref*zeroKernel*Fzero;
    end

    % Scalar sine coefficients for ell>0:
    %
    % [N_s, R_s, G_s, B_s].
    if ell>=1

        basisSin = Bbasis.'*Ssum;

        coeffSin = pref*(Vret.'*basisSin);

    else

        coeffSin = zeros(nr,4);
    end

    coeffAll = [coeffCos coeffSin];

    imagRatio = ...
        norm(imag(coeffAll(:))) / ...
        max(norm(real(coeffAll(:))),eps);

    max_coeff_imag_ratio = ...
        max(max_coeff_imag_ratio,imagRatio);

    time_coeff = time_coeff + toc(tt);

    %% 9C. INDEPENDENT SCALAR PSWF SYNTHESIS

    tt=tic;

    % Push coefficients into the Jacobi/Galerkin basis before spatial
    % evaluation, exactly as in the optimized CPSWF implementation.
    Wcos = Vret*coeffCos;        % m x 4

    if ell>=1
        Wsin = Vret*coeffSin;    % m x 4
    else
        Wsin = zeros(m,4);
    end

    CLunique = local_radial_table_with_rpow( ...
        rUniquePix,rPowUnique,m-1,ell);

    radialCos = CLunique*Wcos;

    cp = real(harmPix);

    if ell==0

        % k=0 has only one real angular basis function.
        Rec = Rec + ...
            cp.*radialCos(pixRadiusGroup,:);

    else

        radialSin = CLunique*Wsin;

        sp = imag(harmPix);

        % Nonzero angular orders contain both +ell and -ell, equivalently
        % the real cosine/sine pair; hence the factor 2.
        Rec = Rec + 2*( ...
            cp.*radialCos(pixRadiusGroup,:) + ...
            sp.*radialSin(pixRadiusGroup,:) );
    end

    time_spatial = time_spatial + toc(tt);

    %% 9D. COEFFICIENT / MODE DIAGNOSTICS

    if ell==0

        mag = abs(coeffCos);

        real_basis_per_channel = ...
            real_basis_per_channel + nr;

    else

        mag = sqrt( ...
            abs(coeffCos).^2 + abs(coeffSin).^2 );

        real_basis_per_channel = ...
            real_basis_per_channel + 2*nr;
    end

    CoeffMagN = [CoeffMagN;mag(:,1)]; %#ok<AGROW>
    CoeffMagR = [CoeffMagR;mag(:,2)]; %#ok<AGROW>
    CoeffMagG = [CoeffMagG;mag(:,3)]; %#ok<AGROW>
    CoeffMagB = [CoeffMagB;mag(:,4)]; %#ok<AGROW>

    stored_nonnegative_modes = ...
        stored_nonnegative_modes + nr;

    ModeEll = [ModeEll;repmat(ell,nr,1)]; %#ok<AGROW>
    ModeN = [ModeN;(1:nr).']; %#ok<AGROW>

    %% 9E. RECURSIVE ANGULAR / RADIAL UPDATE

    harmFreq = harmFreq.*phaseFreq;
    harmPix = harmPix.*phasePix;
    rPowUnique = rPowUnique.*rUniquePix;
end

processing_time = toc(main_timer);

%% 10. FINAL REAL CHANNELS

Rec = real(Rec);

RecN = Rec(:,1);
RecR = Rec(:,2);
RecG = Rec(:,3);
RecB = Rec(:,4);

%% 11. ERROR METRICS

dN = RecN-Nval;
dR = RecR-Rval;
dG = RecG-Gval;
dB = RecB-Bval;

input4 = [Nval;Rval;Gval;Bval];
err4   = [dN;dR;dG;dB];

relative_4ch_error = ...
    norm(err4)/max(norm(input4),eps);

relative_NIR_error = ...
    norm(dN)/max(norm(Nval),eps);

relative_R_error = ...
    norm(dR)/max(norm(Rval),eps);

relative_G_error = ...
    norm(dG)/max(norm(Gval),eps);

relative_B_error = ...
    norm(dB)/max(norm(Bval),eps);

inputRGB = [Rval;Gval;Bval];
errRGB = [dR;dG;dB];

relative_RGB_error = ...
    norm(errRGB)/max(norm(inputRGB),eps);

mse4 = mean(err4.^2);

if mse4>0
    PSNR4 = 10*log10(1/mse4);
else
    PSNR4 = Inf;
end

mseRGB = mean(errRGB.^2);

if mseRGB>0
    PSNR_RGB = 10*log10(1/mseRGB);
else
    PSNR_RGB = Inf;
end

total_real_coefficients = ...
    4*real_basis_per_channel;

%% 12. OPTIONAL SAME-IMAGE CPSWF COMPARISON

cpswf_reference_file = '';
relative_difference_vs_CPSWF = NaN;
max_abs_difference_vs_CPSWF = NaN;

cpswf_error4 = NaN;
cpswf_errorRGB = NaN;
cpswf_errorNIR = NaN;

for q=1:numel(cpswf_reference_candidates)

    candidate = cpswf_reference_candidates{q};

    if isfile(candidate)

        Sref = load(candidate);

        if all(isfield(Sref,{'RecN','RecR','RecG','RecB'}))

            cpswf_reference_file = candidate;

            cpswfVec = [ ...
                Sref.RecN; ...
                Sref.RecR; ...
                Sref.RecG; ...
                Sref.RecB];

            pswfVec = [ ...
                RecN;RecR;RecG;RecB];

            relative_difference_vs_CPSWF = ...
                norm(pswfVec-cpswfVec) / ...
                max(norm(cpswfVec),eps);

            max_abs_difference_vs_CPSWF = ...
                max(abs(pswfVec-cpswfVec));

            if isfield(Sref,'relative_4ch_error')
                cpswf_error4 = Sref.relative_4ch_error;
            end

            if isfield(Sref,'relative_RGB_error')
                cpswf_errorRGB = Sref.relative_RGB_error;
            end

            if isfield(Sref,'relative_NIR_error')
                cpswf_errorNIR = Sref.relative_NIR_error;
            end

            break
        end
    end
end

%% 13. PRINT RESULTS

fprintf('\n================ INDEPENDENT PSWF RESULTS ==================\n');

fprintf('Grid size                         : %d x %d\n',Npix,Npix);
fprintf('Pixels inside disk                : %d\n',Ndisk);
fprintf('Unique spatial radii              : %d\n',NuniquePix);
fprintf('Unique frequency radii            : %d\n',NuniqueFreq);
fprintf('Largest retained ell              : %d\n',lastEll);

fprintf('Stored nonnegative modes/channel  : %d\n', ...
    stored_nonnegative_modes);

fprintf('Real scalar basis funcs/channel   : %d\n', ...
    real_basis_per_channel);

fprintf('Total real coefficients, 4 ch.    : %d\n', ...
    total_real_coefficients);

fprintf('DFT time                          : %.3f s\n',dft_time);
fprintf('Main PSWF reconstruction time     : %.3f s\n',processing_time);

fprintf('  angular grouping                : %.3f s\n',time_angular);
fprintf('  coefficient projection          : %.3f s\n',time_coeff);
fprintf('  spatial synthesis               : %.3f s\n',time_spatial);

fprintf('Combined 4-channel relative error : %.12e\n', ...
    relative_4ch_error);

fprintf('NIR relative error                : %.12e\n', ...
    relative_NIR_error);

fprintf('RGB relative error                : %.12e\n', ...
    relative_RGB_error);

fprintf('RED relative error                : %.12e\n', ...
    relative_R_error);

fprintf('GREEN relative error              : %.12e\n', ...
    relative_G_error);

fprintf('BLUE relative error               : %.12e\n', ...
    relative_B_error);

fprintf('4-channel PSNR                    : %.6f dB\n',PSNR4);
fprintf('RGB PSNR                          : %.6f dB\n',PSNR_RGB);

fprintf('Max coefficient imag/real ratio   : %.3e\n', ...
    max_coeff_imag_ratio);

if ~isempty(cpswf_reference_file)

    fprintf('\nSame-image CPSWF reference        : %s\n', ...
        cpswf_reference_file);

    fprintf('PSWF vs CPSWF relative difference : %.3e\n', ...
        relative_difference_vs_CPSWF);

    fprintf('PSWF vs CPSWF maximum abs diff.   : %.3e\n', ...
        max_abs_difference_vs_CPSWF);

    if isfinite(cpswf_error4)

        fprintf('CPSWF 4-channel error              : %.12e\n', ...
            cpswf_error4);

        fprintf('PSWF-CPSWF 4-ch error difference  : %.3e\n', ...
            relative_4ch_error-cpswf_error4);
    end

    if isfinite(cpswf_errorRGB)

        fprintf('CPSWF RGB error                    : %.12e\n', ...
            cpswf_errorRGB);

        fprintf('PSWF-CPSWF RGB error difference   : %.3e\n', ...
            relative_RGB_error-cpswf_errorRGB);
    end

    if isfinite(cpswf_errorNIR)

        fprintf('CPSWF NIR error                    : %.12e\n', ...
            cpswf_errorNIR);

        fprintf('PSWF-CPSWF NIR error difference   : %.3e\n', ...
            relative_NIR_error-cpswf_errorNIR);
    end

else

    fprintf('\nSame-image CPSWF reference        : not found\n');
end

fprintf('============================================================\n');

%% 14. BUILD DISPLAY ARRAYS

RGB_disk = zeros(Npix,Npix,3);

RGB_disk(:,:,1) = RGB(:,:,1).*inside;
RGB_disk(:,:,2) = RGB(:,:,2).*inside;
RGB_disk(:,:,3) = RGB(:,:,3).*inside;

NIR_disk = NIR.*inside;

RGB_rec = zeros(Npix,Npix,3);

tmp=zeros(Npix,Npix);
tmp(inside)=RecR;
RGB_rec(:,:,1)=tmp;

tmp=zeros(Npix,Npix);
tmp(inside)=RecG;
RGB_rec(:,:,2)=tmp;

tmp=zeros(Npix,Npix);
tmp(inside)=RecB;
RGB_rec(:,:,3)=tmp;

NIR_rec = zeros(Npix,Npix);
NIR_rec(inside)=RecN;

RGB_rec_display = min(max(RGB_rec,0),1);
NIR_rec_display = min(max(NIR_rec,0),1);

RGB_error_image = zeros(Npix,Npix);

RGB_error_image(inside) = ...
    sqrt(dR.^2+dG.^2+dB.^2);

NIR_error_image = zeros(Npix,Npix);
NIR_error_image(inside)=abs(dN);

%% 15. PLOT

figure('Name','Four independent PSWF channels — FFT--Bessel');

subplot(2,4,1)

image(RGB_disk)
axis image off
title('Original RGB')

subplot(2,4,2)

image(RGB_rec_display)
axis image off
title('Independent PSWF RGB')

subplot(2,4,3)

image(repmat(NIR_disk,1,1,3))
axis image off
title('Original NIR')

subplot(2,4,4)

image(repmat(NIR_rec_display,1,1,3))
axis image off
title('Independent PSWF NIR')

subplot(2,4,5)

imagesc(RGB_error_image)
axis image off
colorbar
title('RGB error magnitude')

subplot(2,4,6)

imagesc(NIR_error_image)
axis image off
colorbar
title('|NIR error|')

subplot(2,4,7)

semilogy(CoeffMagN,'LineWidth',0.8)
hold on
semilogy(CoeffMagR,'LineWidth',0.8)
semilogy(CoeffMagG,'LineWidth',0.8)
semilogy(CoeffMagB,'LineWidth',0.8)

grid on

legend('NIR','R','G','B','Location','best')

xlabel('retained nonnegative-order mode index')
title('Scalar PSWF coefficient magnitudes')

subplot(2,4,8)

text(0.02,0.80, ...
    sprintf('4-ch error = %.6e',relative_4ch_error), ...
    'FontSize',11);

text(0.02,0.62, ...
    sprintf('RGB error = %.6e',relative_RGB_error), ...
    'FontSize',11);

text(0.02,0.44, ...
    sprintf('NIR error = %.6e',relative_NIR_error), ...
    'FontSize',11);

if isfinite(relative_difference_vs_CPSWF)

    text(0.02,0.26, ...
        sprintf('PSWF-CPSWF = %.3e', ...
        relative_difference_vs_CPSWF), ...
        'FontSize',11);
end

axis off

title('Numerical comparison')

sgtitle(sprintf( ...
    'Four independent PSWF channels: L=%d, c_{PSWF}=%.3f', ...
    L,c_PSWF));

%% 16. SAVE

imwrite(RGB_rec_display,rgb_output);
imwrite(NIR_rec_display,nir_output);

save(mat_output, ...
    'L','Npix','c_CPSWF','c_PSWF','beta','T','m', ...
    'lastEll','nrByEll', ...
    'stored_nonnegative_modes','real_basis_per_channel', ...
    'total_real_coefficients', ...
    'relative_4ch_error','relative_NIR_error', ...
    'relative_RGB_error','relative_R_error', ...
    'relative_G_error','relative_B_error', ...
    'PSNR4','PSNR_RGB', ...
    'dft_time','processing_time', ...
    'time_angular','time_coeff','time_spatial', ...
    'max_coeff_imag_ratio', ...
    'RecN','RecR','RecG','RecB', ...
    'Nval','Rval','Gval','Bval', ...
    'CoeffMagN','CoeffMagR','CoeffMagG','CoeffMagB', ...
    'ModeEll','ModeN', ...
    'cpswf_reference_file', ...
    'relative_difference_vs_CPSWF', ...
    'max_abs_difference_vs_CPSWF');

fprintf('\nSaved RGB reconstruction : %s\n',rgb_output);
fprintf('Saved NIR reconstruction : %s\n',nir_output);
fprintf('Saved numerical results  : %s\n',mat_output);

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function A=local_to_unit_double(A)

    if isinteger(A)

        A=double(A)/double(intmax(class(A)));

    else

        A=double(A);

        mx=max(A(:));

        if mx>1
            A=A/mx;
        end
    end

    A=min(max(A,0),1);
end

function CL=local_radial_table_with_rpow(r,rPow,Nmax,ell)

    r=r(:);
    rPow=rPow(:);

    P=local_jacobi_alpha0_table( ...
        1-2*r.^2,Nmax,ell);

    n=0:Nmax;

    scale= ...
        sqrt(2*ell+4*n+2)/sqrt(2*pi);

    CL=(rPow.*P).*scale;
end

function P=local_jacobi_alpha0_table(z,Nmax,alpha)

    z=z(:);

    beta=0;

    P=zeros(numel(z),Nmax+1);

    P(:,1)=1;

    if Nmax==0
        return
    end

    P(:,2)= ...
        0.5*((alpha-beta) + ...
        (alpha+beta+2).*z);

    for n=1:Nmax-1

        A1= ...
            2*(n+1)*(n+alpha+beta+1)* ...
            (2*n+alpha+beta);

        A2= ...
            (2*n+alpha+beta+1)* ...
            (alpha^2-beta^2);

        A3= ...
            (2*n+alpha+beta)* ...
            (2*n+alpha+beta+1)* ...
            (2*n+alpha+beta+2);

        A4= ...
            2*(n+alpha)*(n+beta)* ...
            (2*n+alpha+beta+2);

        P(:,n+2)= ...
            ((A2+A3.*z).*P(:,n+1)- ...
             A4.*P(:,n))./A1;
    end
end
