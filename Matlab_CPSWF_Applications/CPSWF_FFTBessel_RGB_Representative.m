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

c_CPSWF = L/2;                   % 64
c_PSWF  = 2*pi*c_CPSWF;          % pi*L
beta    = c_PSWF/(pi*L);         % 1

T = 1e-5;
m = 220;
maxAngular = 520;

rgb_name = '0000_rgb.tiff';

% Derive the image prefix automatically from the RGB filename.
[~,rgb_stem,~] = fileparts(rgb_name);
image_prefix = regexprep(rgb_stem,'_rgb$','');

pad_value = 0;

% Reusable precomputation.  This cache is independent of the image channels.
cache_file = ...
    'CPSWF_FFTBessel_Precompute_L128_c64_m220_T1e-5.mat';

% RGB-only outputs.
rgb_output = sprintf('%s_RGB_CPSWF_FFTBessel_OPT_L128.png',image_prefix);
mat_output = sprintf('%s_RGB_CPSWF_FFTBessel_OPT_L128_results.mat',image_prefix);

% A cheap internal check of the algebraic reordering for the first modes.
verify_reordering = true;
verify_ells = [0 1 2];

alphaM = 2*pi*L/Mgrid;

%% 2. CHECK FILES

addpath(pwd);

if isempty(which('Even_CPSWFs_Matrix'))
    error('Even_CPSWFs_Matrix.m was not found.');
end

if ~isfile(rgb_name)
    error('RGB image %s was not found.',rgb_name);
end

fprintf('\n============================================================\n');
fprintf('OPTIMIZED RGB CPSWF FFT--BESSEL RECONSTRUCTION\n');
fprintf('============================================================\n');
fprintf('RGB image                   : %s\n',rgb_name);
fprintf('Cl_2 input                  : 0 + R e1 + G e2 + B e12\n');
fprintf('Grid                        : %d x %d\n',Npix,Npix);
fprintf('L                           : %d\n',L);
fprintf('CPSWF bandwidth c           : %.12g\n',c_CPSWF);
fprintf('Equivalent PSWF c           : %.12f\n',c_PSWF);
fprintf('alpha_M                     : %.15f\n',alphaM);
fprintf('Omega_T threshold           : %.3e\n',T);
fprintf('Galerkin dimension m        : %d\n',m);
fprintf('Cache                       : %s\n',cache_file);
fprintf('============================================================\n\n');

%% 3. READ AND CROP THE RGB IMAGE

rawRGB = imread(rgb_name);

if ndims(rawRGB) ~= 3 || size(rawRGB,3) < 3
    error('RGB input must contain at least three channels.');
end

RGB0 = local_to_unit_double(rawRGB(:,:,1:3));

origH = size(RGB0,1);
origW = size(RGB0,2);

cx = (origW + 1)/2;
cy = (origH + 1)/2;
circle_radius_px = (min(origH,origW)-1)/2;

x_left   = cx-circle_radius_px;
x_right  = cx+circle_radius_px;
y_top    = cy-circle_radius_px;
y_bottom = cy+circle_radius_px;

[xsrc,ysrc] = meshgrid(1:origW,1:origH);
[xfit,yfit] = meshgrid(linspace(x_left,x_right,Npix), ...
                       linspace(y_top,y_bottom,Npix));

RGB = zeros(Npix,Npix,3);
for ch=1:3
    RGB(:,:,ch) = interp2(xsrc,ysrc,RGB0(:,:,ch), ...
                          xfit,yfit,'linear',pad_value);
end

RGB = min(max(RGB,0),1);

[Xgrid,Ygrid] = meshgrid(-L:L,-L:L);
inside = (Xgrid.^2 + Ygrid.^2) <= L^2;

for ch=1:3
    tmp=RGB(:,:,ch);
    tmp(~inside)=pad_value;
    RGB(:,:,ch)=tmp;
end

x_disk = Xgrid(inside);
y_disk = Ygrid(inside);

r_pix = sqrt(x_disk.^2+y_disk.^2)/L;
theta_pix = atan2(y_disk,x_disk);

Ndisk = nnz(inside);

% Clifford/quaternion embedding:
%   scalar = 0,  e1 = R,  e2 = G,  e12 = B.
ScalarVal = zeros(Ndisk,1);
Rimg = RGB(:,:,1); Rval = Rimg(inside);
Gimg = RGB(:,:,2); Gval = Gimg(inside);
Bimg = RGB(:,:,3); Bval = Bimg(inside);

fprintf('Original RGB size           : %d x %d\n',origH,origW);
fprintf('Pixels inside disk          : %d\n\n',Ndisk);

%% 4. UNIQUE PIXEL RADII
%
% Synthesis is radial, so evaluate radial polynomials only once for each
% distinct integer radius squared x^2+y^2.

pixRadiusSq = x_disk.^2 + y_disk.^2;
[uniquePixRadiusSq,~,pixRadiusGroup] = unique(pixRadiusSq);

rUniquePix = sqrt(uniquePixRadiusSq)/L;
NuniquePix = numel(rUniquePix);

fprintf('Unique spatial radii        : %d (instead of %d pixels)\n\n', ...
        NuniquePix,Ndisk);

%% 5. EXACT CENTERED DFT

ivec = -L:L;

E = exp(-2*pi*1i*(ivec(:)*ivec(:).')/Mgrid);

fprintf('Computing exact centered DFT ...\n');
tt=tic;

% Scalar part is exactly zero.
F0hat  = complex(zeros(Npix,Npix));
F1hat  = E*RGB(:,:,1)*E.';
F2hat  = E*RGB(:,:,2)*E.';
F12hat = E*RGB(:,:,3)*E.';

dft_time=toc(tt);

% Check the centered DFT using the red channel.
R_back = real(E'*F1hat*conj(E))/Mgrid^2;
dft_inverse_error = ...
    norm(R_back(:)-Rimg(:))/max(norm(Rimg(:)),eps);

fprintf('DFT time                    : %.3f s\n',dft_time);
fprintf('Inverse DFT check (R)       : %.3e\n\n',dft_inverse_error);

%% 6. FREQUENCY GEOMETRY AND UNIQUE RADII

[J1,J2] = meshgrid(ivec,ivec);

Jabs = sqrt(J1.^2 + J2.^2);
PhiJ = atan2(J2,J1);

zeroMask = (Jabs==0);
nonzeroMask = ~zeroMask;

J1nz = J1(nonzeroMask);
J2nz = J2(nonzeroMask);
PhiNZ = PhiJ(nonzeroMask);

Ffreq = [ ...
    F0hat(nonzeroMask), ...
    F1hat(nonzeroMask), ...
    F2hat(nonzeroMask), ...
    F12hat(nonzeroMask)];

sqNZ = J1nz.^2 + J2nz.^2;
[uniqueSq,~,freqRadiusGroup] = unique(sqNZ);

rhoUnique = alphaM*sqrt(uniqueSq);
NuniqueFreq = numel(rhoUnique);
Nfreq = numel(freqRadiusGroup);

% Sparse radial grouping operator.  Each frequency belongs to exactly
% one unique-radius group.
RadiusSum = sparse(freqRadiusGroup, ...
                   (1:Nfreq).', ...
                   ones(Nfreq,1), ...
                   NuniqueFreq,Nfreq);

Fzero = [ ...
    F0hat(zeroMask), ...
    F1hat(zeroMask), ...
    F2hat(zeroMask), ...
    F12hat(zeroMask)];

fprintf('Nonzero frequency points    : %d\n',Nfreq);
fprintf('Unique frequency radii      : %d\n\n',NuniqueFreq);

%% 7. LOAD OR BUILD THE REUSABLE PRECOMPUTATION CACHE

cache_ok = false;

if isfile(cache_file)
    fprintf('Loading precomputation cache ...\n');
    tt=tic;
    C = load(cache_file);
    cache_load_time=toc(tt);

    cache_ok = ...
        isfield(C,'cache_L') && C.cache_L==L && ...
        isfield(C,'cache_c') && abs(C.cache_c-c_CPSWF)<1e-14 && ...
        isfield(C,'cache_m') && C.cache_m==m && ...
        isfield(C,'cache_T') && abs(C.cache_T-T)<1e-14 && ...
        isfield(C,'rhoUniqueCache') && ...
        numel(C.rhoUniqueCache)==numel(rhoUnique) && ...
        max(abs(C.rhoUniqueCache(:)-rhoUnique(:)))<1e-12;

    if cache_ok
        VretCell       = C.VretCell;
        nrByEll        = C.nrByEll;
        lastEll        = C.lastEll;
        JoverRho       = C.JoverRho;
        LambdaMagnitude = C.LambdaMagnitude;
        fprintf('Valid cache loaded in %.3f s.\n\n',cache_load_time);
    else
        fprintf('Existing cache is incompatible; rebuilding it.\n\n');
    end
end

if ~cache_ok

    fprintf('============================================================\n');
    fprintf('BUILDING ONE-TIME PRECOMPUTATION CACHE\n');
    fprintf('============================================================\n');

    precompute_timer=tic;

    %--------------------------------------------------------------
    % 7A. Determine the retained radial counts from the Omega_T rule.
    %
    % The truncation depends only on the CPSWF eigenvalues and therefore
    % does not depend on whether the input contains RGB, NIR, or any other
    % Clifford-valued channel data.
    %--------------------------------------------------------------
    if isempty(which('Even_Eigenvalue_CPSWFs'))
        error(['The precomputation cache was not found, and ', ...
               'Even_Eigenvalue_CPSWFs.m is unavailable.']);
    end

    fprintf(['Computing Omega_T mode counts with the CPSWF ', ...
             'eigenvalue routine (one-time cost)...\n']);

    nrTmp = zeros(maxAngular+1,1);
    LambdaMagnitude = [];
    lastEll = -1;

    for ell=0:maxAngular
        mu=zeros(1,m);
        for nEig=1:m
            mu(nEig)=Even_Eigenvalue_CPSWFs(ell,c_CPSWF,m,nEig);
        end

        lambda_mag=abs(c_CPSWF.*mu);
        lambda2=min(lambda_mag.^2,1);
        lambda2g=min(lambda2,1-eps);

        gamma=sqrt(lambda2g ./ max(1-lambda2g,realmin));

        n_end=find(gamma<=T,1,'first');

        if isempty(n_end)
            error('m=%d too small at ell=%d.',m,ell);
        end

        if n_end<2
            break
        end

        nr=n_end-1;
        nrTmp(ell+1)=nr;
        lastEll=ell;

        LambdaMagnitude = ...
            [LambdaMagnitude;lambda_mag(1:nr).']; %#ok<AGROW>

        fprintf('  truncation ell=%3d : nr=%3d\n',ell,nr);
    end

    if lastEll<0
        error('No CPSWF modes were retained for the chosen threshold T.');
    end

    nrByEll=nrTmp(1:lastEll+1);

    %--------------------------------------------------------------
    % 7B. Precompute retained eigenvectors ONCE.
    %--------------------------------------------------------------
    fprintf('\nPrecomputing retained CPSWF eigensystems ...\n');

    VretCell=cell(lastEll+1,1);

    for ell=0:lastEll
        Mmat=Even_CPSWFs_Matrix(ell,c_CPSWF,m);
        Mmat=0.5*(Mmat+Mmat.');

        [V,D]=eig(Mmat);
        [~,idx]=sort(real(diag(D)),'ascend');
        V=real(V(:,idx));

        for j=1:m
            [~,imax]=max(abs(V(:,j)));
            if V(imax,j)<0
                V(:,j)=-V(:,j);
            end
        end

        nr=nrByEll(ell+1);
        VretCell{ell+1}=V(:,1:nr);

        if mod(ell,25)==0 || ell==lastEll
            fprintf('  eigensystems: ell=%3d / %3d\n',ell,lastEll);
        end
    end

    %--------------------------------------------------------------
    % 7C. Precompute every Bessel order that can occur:
    %
    %     order = ell + 2*s + 1,
    %     ell=0,...,lastEll,  s=0,...,m-1.
    %
    % Across all ell, this requires integer orders
    % 1,...,lastEll+2(m-1)+1.
    %--------------------------------------------------------------
    maxBesselOrder = lastEll + 2*(m-1) + 1;

    fprintf('\nPrecomputing J_q(rho)/rho ...\n');
    fprintf('  unique rho values         : %d\n',NuniqueFreq);
    fprintf('  maximum Bessel order      : %d\n',maxBesselOrder);

    JoverRho=zeros(NuniqueFreq,maxBesselOrder);

    for q=1:maxBesselOrder
        JoverRho(:,q)=besselj(q,rhoUnique)./rhoUnique;

        if mod(q,50)==0 || q==maxBesselOrder
            fprintf('  Bessel orders: %4d / %4d\n',q,maxBesselOrder);
        end
    end

    cache_L=L;
    cache_c=c_CPSWF;
    cache_m=m;
    cache_T=T;
    rhoUniqueCache=rhoUnique;

    precompute_time=toc(precompute_timer);

    fprintf('\nSaving reusable cache ...\n');

    save(cache_file, ...
        'cache_L','cache_c','cache_m','cache_T', ...
        'rhoUniqueCache','nrByEll','lastEll','VretCell', ...
        'JoverRho','LambdaMagnitude','precompute_time','-v7.3');

    fprintf('Precomputation finished in %.3f s.\n',precompute_time);
    fprintf('Cache saved to: %s\n',cache_file);
    fprintf('============================================================\n\n');
else
    precompute_time=0;
end

%% 8. INITIALIZE OPTIMIZED RECONSTRUCTION

Rec = complex(zeros(Ndisk,4));   % columns: scalar,R,G,B

CoeffA_mag=[];
CoeffB_mag=[];

ModeEllA=[];
ModeNA=[];
ModeEllB=[];
ModeNB=[];

% Recursive angular harmonics.
phaseFreq=exp(1i*PhiNZ);
harmFreq=ones(Nfreq,1);

phasePix=exp(1i*theta_pix);
harmPix=ones(Ndisk,1);

% Recursive r^ell on the UNIQUE pixel radii.
rPowUnique=ones(NuniquePix,1);

% Timing diagnostics.
time_angular=0;
time_coeff=0;
time_spatial=0;

max_reorder_relerr=0;
max_coeff_imag_ratio=0;

fprintf('Starting optimized FFT--Bessel CPSWF reconstruction ...\n');

main_timer=tic;

%% 9. MAIN LOOP

for ell=0:lastEll

    nr=nrByEll(ell+1);
    Vret=VretCell{ell+1};

    fprintf('  ell=%3d : nr=%3d',ell,nr);
    if ell==0
        fprintf(' (A only)\n');
    else
        fprintf(' (A+B)\n');
    end

    %--------------------------------------------------------------
    % 9A. Angular Fourier grouping by unique radius
    %--------------------------------------------------------------
    tsec=tic;

    Cphi=real(harmFreq);
    Sphi=imag(harmFreq);

    % Columns of Ffreq: scalar(=0),R,G,B.
    Csum=RadiusSum*(Ffreq.*Cphi);
    Ssum=RadiusSum*(Ffreq.*Sphi);

    % Eight right-Cl_2 coefficient integrands:
    %
    % A: [a0,a1,a2,a12]
    % B: [d0,d1,d2,d12]
    Gagg=zeros(NuniqueFreq,8);

    % Family A
    Gagg(:,1)= Csum(:,2)-Ssum(:,3);   % R_c-G_s
    Gagg(:,2)=-Csum(:,1)+Ssum(:,4);   % -S_c+B_s
    Gagg(:,3)= Ssum(:,1)+Csum(:,4);   % S_s+B_c
    Gagg(:,4)=-Ssum(:,2)-Csum(:,3);   % -R_s-G_c

    % Family B
    if ell>=1
        Gagg(:,5)=-Csum(:,1)-Ssum(:,4);  % -S_c-B_s
        Gagg(:,6)=-Csum(:,2)-Ssum(:,3);  % -R_c-G_s
        Gagg(:,7)= Ssum(:,2)-Csum(:,3);  % R_s-G_c
        Gagg(:,8)= Ssum(:,1)-Csum(:,4);  % S_s-B_c
    end

    time_angular=time_angular+toc(tsec);

    %--------------------------------------------------------------
    % 9B. FFT--Bessel radial projection
    %
    % Bbasis(q,s) =
    %   kappa_{s,ell} J_{ell+2s+1}(rho_q)/rho_q.
    %
    % Instead of H=Bbasis*Vret and H'*Gagg, use
    %
    %   Vret'*(Bbasis'*Gagg).
    %--------------------------------------------------------------
    tsec=tic;

    s=0:m-1;
    orders=ell+2*s+1;
    scale=sqrt(2*ell+4*s+2)/sqrt(2*pi);

    Bbasis=JoverRho(:,orders).*scale;

    basisProjection=Bbasis.'*Gagg;   % m x 8

    pref=2*pi*(1i^ell)/Mgrid^2;

    coeff=pref*(Vret.'*basisProjection);   % nr x 8

    % Zero-frequency term occurs only for ell=0 / Family A.
    if ell==0
        zeroKernel=0.5*sqrt(2/(2*pi))*Vret(1,:).';

        coeff(:,1)=coeff(:,1)+pref*zeroKernel*( Fzero(2));
        coeff(:,2)=coeff(:,2)+pref*zeroKernel*(-Fzero(1));
        coeff(:,3)=coeff(:,3)+pref*zeroKernel*( Fzero(4));
        coeff(:,4)=coeff(:,4)+pref*zeroKernel*(-Fzero(3));
    end

    % Optional equivalence check against the original multiplication order.
    if verify_reordering && any(ell==verify_ells)
        Hcheck=Bbasis*Vret;
        coeff_check=pref*(Hcheck.'*Gagg);

        if ell==0
            coeff_check(:,1)=coeff_check(:,1)+pref*zeroKernel*( Fzero(2));
            coeff_check(:,2)=coeff_check(:,2)+pref*zeroKernel*(-Fzero(1));
            coeff_check(:,3)=coeff_check(:,3)+pref*zeroKernel*( Fzero(4));
            coeff_check(:,4)=coeff_check(:,4)+pref*zeroKernel*(-Fzero(3));
        end

        rr=norm(coeff(:)-coeff_check(:))/max(norm(coeff_check(:)),eps);
        max_reorder_relerr=max(max_reorder_relerr,rr);

        fprintf('           reordering check = %.3e\n',rr);
    end

    a0 =coeff(:,1);
    a1 =coeff(:,2);
    a2 =coeff(:,3);
    a12=coeff(:,4);

    if ell>=1
        d0 =coeff(:,5);
        d1 =coeff(:,6);
        d2 =coeff(:,7);
        d12=coeff(:,8);
    else
        d0 =zeros(nr,1);
        d1 =zeros(nr,1);
        d2 =zeros(nr,1);
        d12=zeros(nr,1);
    end

    realNorm=max(norm(real(coeff(:))),eps);
    imagRatio=norm(imag(coeff(:)))/realNorm;
    max_coeff_imag_ratio=max(max_coeff_imag_ratio,imagRatio);

    time_coeff=time_coeff+toc(tsec);

    %--------------------------------------------------------------
    % 9C. Combine A+B algebra BEFORE evaluating the spatial radial basis.
    %
    % For the four Cl_2 components scalar,R,G,B:
    %
    % Rec_scalar = C(-a1-d0) + S(a2+d12)
    % Rec_R = C( a0-d1) + S(-a12+d2)
    % Rec_G = C(-a12-d2)+ S(-a0-d1)
    % Rec_B = C( a2-d12)+ S( a1-d0)
    %--------------------------------------------------------------
    tsec=tic;

    qCos=[ ...
        -a1-d0, ...
         a0-d1, ...
        -a12-d2, ...
         a2-d12];

    qSin=[ ...
         a2+d12, ...
        -a12+d2, ...
        -a0-d1, ...
         a1-d0];

    % Push retained-mode coefficients back to the m-dimensional
    % Jacobi basis BEFORE spatial evaluation.
    Wcos=Vret*qCos;    % m x 4
    Wsin=Vret*qSin;    % m x 4

    % Evaluate Jacobi basis only on UNIQUE pixel radii.
    CLunique=local_radial_table_with_rpow( ...
        rUniquePix,rPowUnique,m-1,ell);

    radialCombined=CLunique*[Wcos Wsin];

    radialCos=radialCombined(:,1:4);
    radialSin=radialCombined(:,5:8);

    cp=real(harmPix);
    sp=imag(harmPix);

    Rec = Rec + ...
        cp.*radialCos(pixRadiusGroup,:) + ...
        sp.*radialSin(pixRadiusGroup,:);

    time_spatial=time_spatial+toc(tsec);

    %--------------------------------------------------------------
    % 9D. Store coefficient diagnostics
    %--------------------------------------------------------------
    CoeffA_mag=[CoeffA_mag; ...
        sqrt(abs(a0).^2+abs(a1).^2+abs(a2).^2+abs(a12).^2)]; %#ok<AGROW>

    ModeEllA=[ModeEllA;repmat(ell,nr,1)]; %#ok<AGROW>
    ModeNA=[ModeNA;(1:nr).']; %#ok<AGROW>

    if ell>=1
        CoeffB_mag=[CoeffB_mag; ...
            sqrt(abs(d0).^2+abs(d1).^2+abs(d2).^2+abs(d12).^2)]; %#ok<AGROW>

        ModeEllB=[ModeEllB;repmat(ell,nr,1)]; %#ok<AGROW>
        ModeNB=[ModeNB;(1:nr).']; %#ok<AGROW>
    end

    % Recursive angular/radial update for next ell.
    harmFreq=harmFreq.*phaseFreq;
    harmPix =harmPix .*phasePix;
    rPowUnique=rPowUnique.*rUniquePix;
end

processing_time=toc(main_timer);

%% 10. FINAL REAL RECONSTRUCTION

RecScalar=real(Rec(:,1));
RecR=real(Rec(:,2));
RecG=real(Rec(:,3));
RecB=real(Rec(:,4));

%% 11. ERROR METRICS

dR=RecR-Rval;
dG=RecG-Gval;
dB=RecB-Bval;

inputRGB=[Rval;Gval;Bval];
errRGB=[dR;dG;dB];

relative_RGB_error=norm(errRGB)/max(norm(inputRGB),eps);
relative_R_error=norm(dR)/max(norm(Rval),eps);
relative_G_error=norm(dG)/max(norm(Gval),eps);
relative_B_error=norm(dB)/max(norm(Bval),eps);

% The input scalar part is identically zero.  Report any scalar component
% produced by the finite CPSWF truncation as a leakage diagnostic.
scalar_leakage_abs=norm(RecScalar);
scalar_leakage_relative_to_RGB=scalar_leakage_abs/max(norm(inputRGB),eps);

mseRGB=mean(errRGB.^2);
if mseRGB>0
    PSNR_RGB=10*log10(1/mseRGB);
else
    PSNR_RGB=Inf;
end

familyA_modes=numel(CoeffA_mag);
familyB_modes=numel(CoeffB_mag);
total_family_modes=familyA_modes+familyB_modes;
total_real_components=4*total_family_modes;

%% 12. RESULTS

fprintf('\n================ OPTIMIZED FFT--BESSEL RESULTS ================\n');
fprintf('Grid size                         : %d x %d\n',Npix,Npix);
fprintf('Pixels inside disk                : %d\n',Ndisk);
fprintf('Unique spatial radii              : %d\n',NuniquePix);
fprintf('Unique frequency radii            : %d\n',NuniqueFreq);
fprintf('Largest retained ell              : %d\n',lastEll);
fprintf('Family A quaternion modes         : %d\n',familyA_modes);
fprintf('Family B quaternion modes         : %d\n',familyB_modes);
fprintf('Total quaternion family modes     : %d\n',total_family_modes);
fprintf('Total real coefficient components : %d\n',total_real_components);
fprintf('DFT time                          : %.3f s\n',dft_time);
fprintf('Main reconstruction time          : %.3f s\n',processing_time);
if exist('cache_load_time','var')
    online_total_time = cache_load_time + dft_time + processing_time;
else
    online_total_time = dft_time + processing_time;
end
fprintf('Online total incl. cache load/DFT : %.3f s\n',online_total_time);
fprintf('  angular grouping                : %.3f s\n',time_angular);
fprintf('  coefficient projection          : %.3f s\n',time_coeff);
fprintf('  spatial synthesis               : %.3f s\n',time_spatial);
fprintf('One-time precompute this run      : %.3f s\n',precompute_time);
fprintf('RGB relative error                : %.12e\n',relative_RGB_error);
fprintf('RED relative error                : %.12e\n',relative_R_error);
fprintf('GREEN relative error              : %.12e\n',relative_G_error);
fprintf('BLUE relative error               : %.12e\n',relative_B_error);
fprintf('RGB PSNR                          : %.6f dB\n',PSNR_RGB);
fprintf('Scalar input norm                 : %.3e\n',norm(ScalarVal));
fprintf('Reconstructed scalar norm         : %.3e\n',scalar_leakage_abs);
fprintf('Scalar leakage / RGB norm         : %.3e\n',scalar_leakage_relative_to_RGB);
fprintf('Max algebraic reorder check       : %.3e\n',max_reorder_relerr);
fprintf('Max coefficient imag/real ratio   : %.3e\n',max_coeff_imag_ratio);
fprintf('================================================================\n');

%% 13. INTERIOR-RADIUS RGB ERROR CHECK

fprintf('\n========== INTERIOR-RADIUS RGB ERROR CHECK ==================\n');

for rcut=[1.00 0.99 0.98 0.95 0.90]
    keep=(r_pix<=rcut);

    e=[RecR(keep)-Rval(keep); ...
       RecG(keep)-Gval(keep); ...
       RecB(keep)-Bval(keep)];

    f=[Rval(keep);Gval(keep);Bval(keep)];

    err=norm(e)/max(norm(f),eps);

    fprintf('r <= %.2f : relative RGB error = %.6e\n',rcut,err);
end

fprintf('============================================================\n');

%% 14. BUILD DISPLAY ARRAYS

RGB_disk=zeros(Npix,Npix,3);
RGB_disk(:,:,1)=RGB(:,:,1).*inside;
RGB_disk(:,:,2)=RGB(:,:,2).*inside;
RGB_disk(:,:,3)=RGB(:,:,3).*inside;

RGB_rec=zeros(Npix,Npix,3);
tmp=zeros(Npix,Npix); tmp(inside)=RecR; RGB_rec(:,:,1)=tmp;
tmp=zeros(Npix,Npix); tmp(inside)=RecG; RGB_rec(:,:,2)=tmp;
tmp=zeros(Npix,Npix); tmp(inside)=RecB; RGB_rec(:,:,3)=tmp;

RGB_rec_display=min(max(RGB_rec,0),1);

RGB_error_image=zeros(Npix,Npix);
RGB_error_image(inside)=sqrt(dR.^2+dG.^2+dB.^2);

Scalar_leakage_image=zeros(Npix,Npix);
Scalar_leakage_image(inside)=abs(RecScalar);

%% 15. REPRESENTATIVE RGB PAPER FIGURE
%
% Top row: Original RGB | CPSWF RGB reconstruction
% Bottom row: RGB pointwise error | scalar-leakage magnitude
%
% The scalar-leakage panel is included only as a diagnostic.  The input
% scalar component is identically zero.

figRep = figure( ...
    'Name','Representative RGB CPSWF FFT--Bessel reconstruction', ...
    'Color','w', ...
    'Units','pixels', ...
    'Position',[80 60 1100 850]);

annotation(figRep,'textbox',[0.06 0.945 0.88 0.035], ...
    'String','RGB CPSWF FFT--Bessel reconstruction', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'FontWeight','bold', ...
    'FontSize',16);

annotation(figRep,'textbox',[0.06 0.910 0.88 0.030], ...
    'String',sprintf(['L = %d,   c_{CPSWF} = %g,   ', ...
                      'RGB relative error = %.3e'], ...
                      L,c_CPSWF,relative_RGB_error), ...
    'Interpreter','tex', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'FontSize',12);

ax1 = axes(figRep,'Position',[0.08 0.525 0.37 0.34]);
image(ax1,RGB_disk)
axis(ax1,'image')
axis(ax1,'off')
title(ax1,'Original RGB','FontWeight','bold','FontSize',11)

ax2 = axes(figRep,'Position',[0.55 0.525 0.37 0.34]);
image(ax2,RGB_rec_display)
axis(ax2,'image')
axis(ax2,'off')
title(ax2,'CPSWF FFT--Bessel RGB','FontWeight','bold','FontSize',11)

ax3 = axes(figRep,'Position',[0.12 0.085 0.31 0.31]);
imagesc(ax3,RGB_error_image)
axis(ax3,'image')
axis(ax3,'off')
title(ax3,'Pointwise RGB error magnitude', ...
      'FontWeight','bold','FontSize',11)
colormap(ax3,parula(256))
cb1=colorbar(ax3);
cb1.FontSize=9;

ax4 = axes(figRep,'Position',[0.59 0.085 0.31 0.31]);
imagesc(ax4,Scalar_leakage_image)
axis(ax4,'image')
axis(ax4,'off')
title(ax4,'Scalar-part leakage (input scalar = 0)', ...
      'FontWeight','bold','FontSize',11)
colormap(ax4,parula(256))
cb2=colorbar(ax4);
cb2.FontSize=9;

drawnow;

representative_output = 'CPSWF_FFTBessel_RGB_Representative.png';
set(figRep,'PaperPositionMode','auto');
print(figRep,representative_output,'-dpng','-r300');

fprintf('Saved representative figure: %s\n',representative_output);

%% 16. SAVE

imwrite(RGB_rec_display,rgb_output);

save(mat_output, ...
    'L','Npix','c_CPSWF','c_PSWF','beta','T','m', ...
    'alphaM','lastEll','nrByEll', ...
    'familyA_modes','familyB_modes','total_family_modes', ...
    'total_real_components','relative_RGB_error', ...
    'relative_R_error','relative_G_error','relative_B_error', ...
    'PSNR_RGB','dft_time','processing_time', ...
    'precompute_time','time_angular','time_coeff','time_spatial', ...
    'scalar_leakage_abs','scalar_leakage_relative_to_RGB', ...
    'max_reorder_relerr','max_coeff_imag_ratio', ...
    'RecScalar','RecR','RecG','RecB', ...
    'ScalarVal','Rval','Gval','Bval', ...
    'CoeffA_mag','CoeffB_mag', ...
    'ModeEllA','ModeNA','ModeEllB','ModeNB', ...
    'LambdaMagnitude');

fprintf('\nSaved RGB reconstruction : %s\n',rgb_output);
fprintf('Saved numerical results  : %s\n',mat_output);
fprintf('Reusable cache           : %s\n',cache_file);

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
%
% Same normalized radial Clifford--Legendre table used in the validated
% scripts, but r^ell is supplied recursively.

    r=r(:);
    rPow=rPow(:);

    P=local_jacobi_alpha0_table(1-2*r.^2,Nmax,ell);

    n=0:Nmax;
    scale=sqrt(2*ell+4*n+2)/sqrt(2*pi);

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

    P(:,2)=0.5*((alpha-beta)+(alpha+beta+2).*z);

    for n=1:Nmax-1

        A1=2*(n+1)*(n+alpha+beta+1)*(2*n+alpha+beta);

        A2=(2*n+alpha+beta+1)*(alpha^2-beta^2);

        A3=(2*n+alpha+beta)*(2*n+alpha+beta+1)* ...
           (2*n+alpha+beta+2);

        A4=2*(n+alpha)*(n+beta)*(2*n+alpha+beta+2);

        P(:,n+2)=((A2+A3.*z).*P(:,n+1)- ...
                  A4.*P(:,n))./A1;
    end
end
