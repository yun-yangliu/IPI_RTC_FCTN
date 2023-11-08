clear;
close all;
addpath(genpath('lib'));
addpath(genpath('RTC_FCTN'));

%% Load initial data
load('HSV_test.mat')
if max(X(:))>1
    X = X/max(X(:));
end

%% Generate observed data
sample_ratio = 0.3;
fprintf('=== The sample ratio is %4.2f ===\n', sample_ratio);
T         = X;
Ndim      = ndims(T);
Nway      = size(T);
for i=1:Nway(3)
    for j=1:Nway(4)
        Nhsi(:,:,i,j) = imnoise(T(:,:,i,j),'salt & pepper',0.1);
    end
end
Omega     = find(rand(prod(Nway),1)<sample_ratio);
F         = zeros(Nway);
F(Omega)  = Nhsi(Omega);

%% Perform RC_FCTN
Ind  = zeros(Nway);
Ind(Omega)  = 1;
for lamb = [14]
    for gamma = [1e-4]
        for deta = [0.0007]
            for f = [0.8]
                lambda = lamb/sqrt(max(Nway(1),Nway(2))*Nway(3)*Nway(4));
                opts.gamma = gamma;
                opts.tol = 1e-4;
                opts.deta = deta;
                opts.maxit = 150;
                opts.f = f;
                opts.Xtrue = T;
                t0=tic;
                [Re_tensor, S, Out,iter] = RC_FCTN(F, lambda, Ind, opts);
                time=toc(t0)
                [psnr,ssim]=MSIQA(T*255, Re_tensor*255)
            end
        end
    end
end
