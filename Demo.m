clear;
close all;
clc;	
addpath(genpath([cd,'\']));
%% simulated experiment
% Case 1 Periodic stripes r = 0.2, I = 0.2
load Per_r2_I2
% Case 2 NonPeriodic stripes r = 0.2, I = 0.2
% load NonPer_r2_I2
load simu_DC
tsize = size(Is);
%% parameters
% different images and stripe noise levels may need change parameters slightly.
opts.lambda1 = 0.0005;               
opts.lambda2 = 0.001;                
opts.lambda3 = 0.01;                 
opts.beta   = 0.1;
opts.maxit   = 1000;
opts.rk   = [1 tsize(3) tsize(3)];
opts.tol     = 5e-5;
%% 
disp('Begin  algorithm')
[X_re,S_re] = LRTD_destriping(Is,opts);
%% 
[psnr, ssim, msam] = MSIQA(simu_DC*255, X_re*255);
fprintf('Periodic case (r=0.2,I=0.2):\n MPSNR = %f, MSSIM = %f, MSAM = %f\n',psnr,ssim,msam);
%fprintf('NonPeriodic case (r=0.2,I=0.2):\n MPSNR = %f, MSSIM = %f, MSAM = %f\n',psnr,ssim,msam);
figure
subplot(131),imshow(simu_DC(:,:,5)),title('Original')
subplot(132),imshow(Is(:,:,5)),title('Striped')
subplot(133),imshow(X_re(:,:,5)),title('LRTD')




