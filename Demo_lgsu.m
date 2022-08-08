clear all
close all
clc
%[Y,XT,A,nc,nr,nb]=Syn(50);

%%
for SNR=50
    if SNR==30
        load DC2_30dB.mat      
        sw=3;lambda_glo=1e-1;lambda_loc=1e-3;
        
    elseif SNR==40
        
        load DC2_40dB.mat        
        sw=3;lambda_glo=5e-2;lambda_loc=1e-4;
    elseif SNR==50
        
        load DC2_50dB.mat        
        sw=3;lambda_glo=1e-2;lambda_loc=1e-5;
    end
    
    
    %%
    
    tic
    Sw = sw; P = round(nr*nc/Sw^2); Ws = log10(sqrt(SNR/3));
    seg = slic3Dhsi(X, P, Ws);
    parameter.lambda_s = lambda_glo;
    parameter.lambda_p = lambda_loc;
    parameter.epsilon = 1e-5;
    parameter.maxiter = 200;
    parameter.mu = 0.20;
    parameter.xt = XT;
    parameter.verbose = 1;
    parameter.seg=seg;
    parameter.imgsize=[nr,nc];
    X_lgsu = lgsuGithub(Y, A, parameter);
    toc
    SRE_lgsu = 20*log10(norm(XT,'fro')/norm(X_lgsu-XT,'fro'));
    SPA_lgsu=length(find(X_lgsu>0.005))/((size(X_lgsu,1)*size(X_lgsu,2)));
    %
    RMSE_lgsu=sqrt(mean2((X_lgsu-XT).^2));
    
    
    %%
    
    figure
    p=9;
    for j=1:p
        subplot_tight(2, p, j,[.003 .003]);
        imagesc(reshape(XT(j,:)',nr, nc),[0,1]);axis image;axis off;
        subplot_tight(2, p, j+p,[.003 .003]);
        imagesc(reshape(X_lgsu(j,:)',nr, nc),[0,1]);axis image;axis off;
    end
    drawnow;
    
    
    
end



