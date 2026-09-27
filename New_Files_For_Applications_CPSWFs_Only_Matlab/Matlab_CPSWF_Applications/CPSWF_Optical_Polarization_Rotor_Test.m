% References
%
% 1. Ghaffari, H. B., Hogan, J. A., & Lakey, J. D. (2022).
%    Properties of Clifford-Legendre Polynomials.
%    Advances in Applied Clifford Algebras, 32(1), 1-25.
%    https://doi.org/10.1007/s00006-021-01179-8
%
% 2. H. Baghal Ghaffari, "Higher-dimensional prolate spheroidal wave
%    functions," Ph.D. dissertation, The University of Newcastle, 2022.
clear; close all; clc;

%% 1. Parameters
L=64; Npix=2*L+1;
c_CPSWF=L/2; c_PSWF=2*pi*c_CPSWF;
T=1e-5; mGalerkin=160; Nr=200; Ntheta=768;
orders=[1,3];
phi1=deg2rad(35); phi3=deg2rad(-50);

addpath(pwd);
assert(~isempty(which('Even_CPSWFs_Matrix')),'Even_CPSWFs_Matrix.m missing');
assert(~isempty(which('Even_Eigenvalue_CPSWFs')),'Even_Eigenvalue_CPSWFs.m missing');
assert(~isempty(which('lgwt')),'lgwt.m missing');

fprintf('\n============================================================\n');
fprintf('OPTICAL CPSWF POLARIZATION-ROTOR TEST\n');
fprintf('============================================================\n');
fprintf('Grid                    : %d x %d\n',Npix,Npix);
fprintf('c_CPSWF                 : %.12g\n',c_CPSWF);
fprintf('c_PSWF                  : %.12g\n',c_PSWF);
fprintf('T                       : %.3e\n',T);
fprintf('Galerkin dimension      : %d\n',mGalerkin);
fprintf('Nr, Ntheta              : %d, %d\n',Nr,Ntheta);
fprintf('k=1 rotor               : +35 deg\n');
fprintf('k=3 rotor               : -50 deg\n');
fprintf('============================================================\n\n');

%% 2. Synthetic complex vector optical field
xvec=linspace(-1,1,Npix); [x,y]=meshgrid(xvec,xvec);
r=sqrt(x.^2+y.^2); th=atan2(y,x); disk=(r<=1);
A1=r.*exp(-3.5*r.^2);
A3=r.^3.*exp(-2.5*r.^2);
a3=0.80; delta=0.85; phase3=a3*exp(1i*delta);

% k=1 radial-like component
Ex1=A1.*cos(th); Ey1=A1.*sin(th);
% k=3 azimuthal-like component
Ex3=-phase3*A3.*sin(3*th); Ey3=phase3*A3.*cos(3*th);

Ex=Ex1+Ex3; Ey=Ey1+Ey3;
Ex(~disk)=0; Ey(~disk)=0;

%% 3. Exact target after order-dependent polarization rotation
c1=cos(phi1); s1=sin(phi1); c3=cos(phi3); s3=sin(phi3);
Ex1_t=c1*Ex1+s1*Ey1;  Ey1_t=-s1*Ex1+c1*Ey1;
Ex3_t=c3*Ex3+s3*Ey3;  Ey3_t=-s3*Ex3+c3*Ey3;
ExT=Ex1_t+Ex3_t; EyT=Ey1_t+Ey3_t;
ExT(~disk)=0; EyT(~disk)=0;

rpix=r(disk); thpix=th(disk);
Exd=Ex(disk); Eyd=Ey(disk); ExTd=ExT(disk); EyTd=EyT(disk);
Ndisk=nnz(disk);
fprintf('Pixels inside disk      : %d\n\n',Ndisk);

%% 4. Polar quadrature of original field
[rq,wr]=lgwt(Nr,0,1); rq=rq(:); wr=wr(:);
thq=(2*pi/Ntheta)*(0:Ntheta-1); dth=2*pi/Ntheta;
[THQ,RQ]=meshgrid(thq,rq);
A1Q=RQ.*exp(-3.5*RQ.^2); A3Q=RQ.^3.*exp(-2.5*RQ.^2);
ExQ=A1Q.*cos(THQ)-phase3*A3Q.*sin(3*THQ);
EyQ=A1Q.*sin(THQ)+phase3*A3Q.*cos(3*THQ);
rw=wr.*rq;

%% 5. Reconstructions
ExC=complex(zeros(Ndisk,1)); EyC=ExC;          % Clifford/CPSWF rotor
ExG=complex(zeros(Ndisk,1)); EyG=ExG;          % coupled scalar PSWF
ExI=complex(zeros(Ndisk,1)); EyI=ExI;          % independent scalar PSWF
Ex0=complex(zeros(Ndisk,1)); Ey0=Ex0;          % original sanity check
best_hx=zeros(2,1); best_hy=zeros(2,1);
Acount=0; Bcount=0;

tic
for io=1:numel(orders)
    k=orders(io);
    if k==1, phi=phi1; else, phi=phi3; end
    cr=cos(phi); sr=sin(phi);

    M=Even_CPSWFs_Matrix(k,c_CPSWF,mGalerkin); M=0.5*(M+M.');
    [V,D]=eig(M); [~,idx]=sort(real(diag(D)),'ascend'); V=real(V(:,idx));
    for j=1:mGalerkin
        [~,im]=max(abs(V(:,j))); if V(im,j)<0, V(:,j)=-V(:,j); end
    end

    mu=zeros(1,mGalerkin);
    for j=1:mGalerkin
        mu(j)=Even_Eigenvalue_CPSWFs(k,c_CPSWF,mGalerkin,j);
    end
    lm=abs(c_CPSWF.*mu); l2=min(lm.^2,1); l2g=min(l2,1-eps);
    gamma=sqrt(l2g./max(1-l2g,realmin)); nend=find(gamma<=T,1,'first');
    if isempty(nend)||nend<2, error('No retained radial set for k=%d',k); end
    nr=nend-1; Vret=V(:,1:nr); Acount=Acount+nr; Bcount=Bcount+nr;
    fprintf('k=%d: %d radial modes, rotor %+g deg\n',k,nr,rad2deg(phi));

    Rqtab=local_radial_table(rq,mGalerkin-1,k)*Vret;
    Rptab=local_radial_table(rpix,mGalerkin-1,k)*Vret;

    ckq=cos(k*thq).'; skq=sin(k*thq).';
    Exc=Rqtab.'*(rw.*(dth*(ExQ*ckq))); Exs=Rqtab.'*(rw.*(dth*(ExQ*skq)));
    Eyc=Rqtab.'*(rw.*(dth*(EyQ*ckq))); Eys=Rqtab.'*(rw.*(dth*(EyQ*skq)));

    C=bsxfun(@times,Rptab,cos(k*thpix));
    S=bsxfun(@times,Rptab,sin(k*thpix));

    % Original reconstruction
    Ex0=Ex0+2*(C*Exc+S*Exs); Ey0=Ey0+2*(C*Eyc+S*Eys);

    % A) CPSWF/Clifford coefficients for F=Ex e1 + Ey e2
    a0=Exc-Eys; a12=-Exs-Eyc;
    d1=-Exc-Eys; d2=Exs-Eyc;
    % right rotor R=cr+sr e12
    a0R=cr*a0-sr*a12; a12R=sr*a0+cr*a12;
    d1R=cr*d1+sr*d2; d2R=-sr*d1+cr*d2;
    ExC=ExC + C*a0R-S*a12R-C*d1R+S*d2R;
    EyC=EyC - C*a12R-S*a0R-C*d2R-S*d1R;

    % B) coupled scalar PSWF (explicit 2x2 rotation)
    ExcT=cr*Exc+sr*Eyc; ExsT=cr*Exs+sr*Eys;
    EycT=-sr*Exc+cr*Eyc; EysT=-sr*Exs+cr*Eys;
    ExG=ExG+2*(C*ExcT+S*ExsT); EyG=EyG+2*(C*EycT+S*EysT);

    % C) best independent scalar multiplier per channel/order
    xs=[Exc;Exs]; xt=[ExcT;ExsT]; ys=[Eyc;Eys]; yt=[EycT;EysT];
    hx=(xs'*xt)/max(xs'*xs,eps); hy=(ys'*yt)/max(ys'*ys,eps);
    best_hx(io)=hx; best_hy(io)=hy;
    ExI=ExI+2*(C*(hx*Exc)+S*(hx*Exs));
    EyI=EyI+2*(C*(hy*Eyc)+S*(hy*Eys));
end
runtime=toc;

%% 6. Errors
normT=sqrt(norm(ExTd)^2+norm(EyTd)^2);
origErr=sqrt(norm(Ex0-Exd)^2+norm(Ey0-Eyd)^2)/sqrt(norm(Exd)^2+norm(Eyd)^2);
errC=sqrt(norm(ExC-ExTd)^2+norm(EyC-EyTd)^2)/normT;
errG=sqrt(norm(ExG-ExTd)^2+norm(EyG-EyTd)^2)/normT;
errI=sqrt(norm(ExI-ExTd)^2+norm(EyI-EyTd)^2)/normT;
diffCG=sqrt(norm(ExC-ExG)^2+norm(EyC-EyG)^2)/normT;

[S0t,S1t,S2t,S3t]=local_stokes(ExTd,EyTd);
[S0c,S1c,S2c,S3c]=local_stokes(ExC,EyC);
[S0g,S1g,S2g,S3g]=local_stokes(ExG,EyG);
[S0i,S1i,S2i,S3i]=local_stokes(ExI,EyI);
ST=[S0t;S1t;S2t;S3t];
stC=norm([S0c;S1c;S2c;S3c]-ST)/norm(ST);
stG=norm([S0g;S1g;S2g;S3g]-ST)/norm(ST);
stI=norm([S0i;S1i;S2i;S3i]-ST)/norm(ST);

fprintf('\n============================================================\n');
fprintf('OPTICAL POLARIZATION-ROTOR RESULTS\n');
fprintf('============================================================\n');
fprintf('Original reconstruction error             : %.12e\n',origErr);
fprintf('Runtime                                   : %.3f s\n',runtime);
fprintf('Family A/B modes                          : %d / %d\n',Acount,Bcount);
fprintf('\nTarget-field relative error:\n');
fprintf('  CPSWF + Clifford rotor                  : %.12e\n',errC);
fprintf('  Coupled scalar PSWF                    : %.12e\n',errG);
fprintf('  Best independent scalar PSWF           : %.12e\n',errI);
fprintf('CPSWF vs coupled PSWF difference         : %.12e\n',diffCG);
fprintf('\n4-Stokes relative error:\n');
fprintf('  CPSWF + Clifford rotor                  : %.12e\n',stC);
fprintf('  Coupled scalar PSWF                    : %.12e\n',stG);
fprintf('  Best independent scalar PSWF           : %.12e\n',stI);
for io=1:2
    fprintf('k=%d independent gains: hx=%+.6f%+.6fi, hy=%+.6f%+.6fi\n',orders(io),real(best_hx(io)),imag(best_hx(io)),real(best_hy(io)),imag(best_hy(io)));
end
fprintf('============================================================\n');

%% 7. Figures
ExTg=local_to_grid(ExTd,disk,Npix); EyTg=local_to_grid(EyTd,disk,Npix);
ExCg=local_to_grid(ExC,disk,Npix);  EyCg=local_to_grid(EyC,disk,Npix);
ExGg=local_to_grid(ExG,disk,Npix);  EyGg=local_to_grid(EyG,disk,Npix);
ExIg=local_to_grid(ExI,disk,Npix);  EyIg=local_to_grid(EyI,disk,Npix);

figure('Name','Optical polarization rotor comparison');
A={ExTg,ExCg,ExGg,ExIg,EyTg,EyCg,EyGg,EyIg};
Ttl={'Target Re(E_x)','CPSWF rotor','Coupled PSWF','Independent PSWF', ...
     'Target Re(E_y)','CPSWF rotor','Coupled PSWF','Independent PSWF'};
for j=1:8
    subplot(2,4,j); tmp=real(A{j}); tmp(~disk)=NaN; imagesc(xvec,xvec,tmp); axis image xy; title(Ttl{j}); colorbar;
end
sgtitle(sprintf('Mode-dependent polarization rotation: CPSWF %.2e, coupled PSWF %.2e, independent %.2e',errC,errG,errI));

figure('Name','Stokes error comparison');
bar([stC,stG,stI]);
set(gca,'XTickLabel',{'CPSWF rotor','Coupled PSWF','Independent PSWF'});
ylabel('relative 4-Stokes error'); title('Polarization-state error'); grid on;

save('CPSWF_Optical_Polarization_Rotor_results.mat','errC','errG','errI','diffCG','stC','stG','stI','origErr','best_hx','best_hy','orders','runtime');
fprintf('\nSaved: CPSWF_Optical_Polarization_Rotor_results.mat\n');

%% Local functions
function CL=local_radial_table(r,Nmax,k)
    r=r(:); z=1-2*r.^2; P=local_jacobi_table(z,Nmax,k,0);
    N=0:Nmax; sc=sqrt(2*k+4*N+2)/sqrt(2*pi);
    CL=bsxfun(@times,r.^k,P); CL=bsxfun(@times,CL,sc);
end

function P=local_jacobi_table(z,Nmax,a,b)
    z=z(:); P=zeros(numel(z),Nmax+1); P(:,1)=1; if Nmax==0, return; end
    P(:,2)=0.5*((a-b)+(a+b+2).*z);
    for n=1:Nmax-1
        A1=2*(n+1)*(n+a+b+1)*(2*n+a+b);
        A2=(2*n+a+b+1)*(a^2-b^2);
        A3=(2*n+a+b)*(2*n+a+b+1)*(2*n+a+b+2);
        A4=2*(n+a)*(n+b)*(2*n+a+b+2);
        P(:,n+2)=((A2+A3.*z).*P(:,n+1)-A4.*P(:,n))./A1;
    end
end

function [S0,S1,S2,S3]=local_stokes(Ex,Ey)
    S0=abs(Ex).^2+abs(Ey).^2; S1=abs(Ex).^2-abs(Ey).^2;
    S2=2*real(Ex.*conj(Ey)); S3=-2*imag(Ex.*conj(Ey));
end

function A=local_to_grid(v,disk,Npix)
    A=complex(zeros(Npix,Npix)); A(disk)=v;
end
