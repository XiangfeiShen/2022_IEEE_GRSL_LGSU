function [X] = lgsuGithub(Y, A, parameter)
% Superpixel-guided Local Sparsity Prior for Hyperspectral Sparse Regression Unmixing
% 
% [LGSU] Xiangfei Shen, Haijun Liu, Xinzheng Zhang, Kai Qin, and Xichuan Zhou*,
%"Superpixel-guided Local Sparsity Prior for Hyperspectral Sparse Regression Unmixing", 
% IEEE Geoscience and Remote Sensing Letters, in Peer Review.
%
% -------------------------------------------------------------------
%
% Usage:
%
% X = lgsu(Y, A, parameter)
%
% ------- Input variables -------------------------------------------
%
%  Y - hyperspectral data matrix with dimensions L(bands) x K(pixels)
%
%  A - spectral library matrix with dimentsions L(bands) x m(spectra)
%
%  parameter.
%            * lambda_p - scalar, regularization parameter of local sparsity 
%            * lambda_s - scalar, regularization parameter of global sparsitye  
%            mu - scalar, initial augmented Lagrangian penalty parameter, 
%              default: 0.2
%            seg - superpixel segmentation
%            epsilon - scalar, scaled error tolerance, default: 1e-5
%            maxiter - scalar, maximum iteration number, default: 500
%            verbose - output the process (1) or not (0), default: 1
%
%  NOTE: PARAMETERS WITH SYMBOL * ARE NECESSARY
%
% ------- Output variables -------------------------------------------
%
% X - the estimated abundance matrix with dimensions m(spectra) x K(pixels)
%
% ---------------------------------------------------------------------
%
% Please see [LGSU] for more details.
%
% Please contact Xiangfei Shen (xfshen95@163.com) to report bugs or 
% provide suggestions and discussions for the codes.
%
% ---------------------------------------------------------------------
% version: 1.0 (27-Jul-2022)
% ---------------------------------------------------------------------
%
% Copyright (Jul, 2022):       Xiangfei Shen (xfshen95@163.com/xfshen95@outlook.com)
%                              Xichuan Zhou (zxc@cqu.edu.cn)
%
% LGSU is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ---------------------------------------------------------------------

%%
%--------------------------------------------------------------
% load and test required parameters
%--------------------------------------------------------------
[L, K] = size(Y);
m = size(A, 2);
if size(A, 1) ~= L;error(['The sizes of hyperspectral data matrix Y and spectral ','library matrix A are inconsistent!']);end
if isfield(parameter, 'maxiter');MaxIter = parameter.maxiter;else MaxIter = 500;end
if isfield(parameter, 'epsilon');epsilon = parameter.epsilon;else epsilon = 1e-5;end
if isfield(parameter, 'epsilon');XT = parameter.xt;else error('The GT is not available'); end
if isfield(parameter, 'mu'); mu = parameter.mu;else mu = 0.1;end
if isfield(parameter, 'lambda_p');lambda1 = parameter.lambda_p;else error('The parameter lambda_p is missing!');end
if isfield(parameter, 'seg');seg = parameter.seg;else error('The parameter seg is missing!');end
if isfield(parameter, 'imgsize');imgsize = parameter.imgsize;else error('The parameter lambda_s is missing!');end
if isfield(parameter, 'lambda_s');lambda2 = parameter.lambda_s;else error('The parameter lambda_s is missing!');end
if isfield(parameter, 'verbose');verbose = parameter.verbose;else verbose = 1;end
if verbose==1;figure; end

Cj=seg.Cj;
labels=seg.labels;

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------


IF = (A'*A + 3*eye(m))^-1;
U = IF*A'*Y;

V1 = A*U;
V2 = U;
V3 = U;
V4 = U;

D1 = V1*0;
D2 = V2*0;
D3 = V3*0;
D4 = V4*0;

%current iteration number
i = 1;

%primal residual 
res_p = inf;

%dual residual
res_d = inf;

%error tolerance
tol = sqrt((3*m + L)/2*K/2)*epsilon;


%%
%---------------------------------------------
%  ADMM iterations
%---------------------------------------------
while (i <= MaxIter) && ((abs(res_p) > tol) || (abs(res_d) > tol))
    if mod(i, 10) == 1
        V10 = V1;
        V20 = V2;
        V30 = V3;
        V40 = V4;
    end
    %update U and V
    U = IF*(A'*(V1 + D1) + (V2 + D2) + (V3 + D3) + (V4 + D4));
    V1 = 1/(1+mu)*(Y + mu*(A*U - D1));
    Au=U - D2;
    for jj =1:seg.P 
        W1 = repmat(1./(sqrt(sum(Au(:,labels==jj).^2,2))+eps),1, size(Au(:,labels==jj),2));
        cj=mean(Cj(labels==jj));
        V2(:,labels==jj) = soft(Au(:,labels==jj), W1*cj*lambda1/mu);
    end
    W2 = SSpasity(U-D3,imgsize);
    V3 = soft(U - D3, W2.*lambda2/mu);
    V4 = max(U - D4, 0);
    %update D
    D1 = D1 - A*U + V1;
    D2 = D2 - U + V2;
    D3 = D3 - U + V3;
    D4 = D4 - U + V4;
    
    if mod(i, 10) == 1
        %object function
        obj = 1/2*norm(A*U - Y, 'fro');
        %primal residual
        res_p = norm([V1; V2; V3; V4] - [A*U; U; U; U], 'fro');
        %dual residual
        res_d = norm([V1; V2; V3; V4] - [V10; V20; V30; V40], 'fro');
  
        if res_p > 10*res_d
            mu = mu*2;
            D1 = D1/2;
            D2 = D2/2;
            D3 = D3/2;
            D4 = D4/2;
        elseif res_d > 10*res_p
            mu = mu/2;
            D1 = D1*2;
            D2 = D2*2;
            D3 = D3*2;
            D4 = D4*2;
        end
    end
    sre(i)=20*log10(norm(XT,'fro')/norm(U-XT,'fro'));
    xreds(i)=norm(U-XT,'fro');
    obj1(i) = norm(A*U - Y, 'fro');
    %primal residual
    res_p1(i) = norm([V1; V2; V3; V4] - [A*U; U; U; U], 'fro');
    %dual residual
     res_d1(i) = norm([V1; V2; V3; V4] - [V10; V20; V30; V40], 'fro');


    if verbose==1
    subplot_tight(5, 3, 1,[.08 .08]); plot(obj1,'-ro'); xlim([0 MaxIter]); title('Objective Values','fontsize',8);
    subplot_tight(5, 3, 2,[.08 .08]); plot(res_p1,'-r.');hold on;plot(res_d1,'-g.'); xlim([0 MaxIter]); title('Residuals','fontsize',8);
    subplot_tight(5, 3, 3,[.08 .08]); plot(sre,'-ro'); xlim([0 MaxIter]); title('SRE Values','fontsize',8);
  
    subplot_tight(5, 3, 4,[.08 .08]); imagesc(XT); xlim([0 MaxIter]); title('XT','fontsize',8);axis off
    subplot_tight(5, 3, 5,[.08 .08]); imagesc(V3); xlim([0 MaxIter]); title('Global Pers.','fontsize',8);axis off
    subplot_tight(5, 3, 6,[.08 .08]); imagesc(V2); xlim([0 MaxIter]); title('Local Pers.','fontsize',8);axis off

    subplot_tight(5, 3, 7,[.08 .08]); imagesc(reshape(U(2,:),imgsize)); title('Abundance #2','fontsize',8);axis off
    subplot_tight(5, 3, 8,[.08 .08]); imagesc(reshape(XT(2,:),imgsize)); title('GT #2','fontsize',8);axis off
    subplot_tight(5, 3, 9,[.08 .08]); imagesc(abs(reshape(XT(2,:),imgsize)-reshape(U(2,:),imgsize))); title('Diff #2','fontsize',8);axis off


    subplot_tight(5, 3, 10,[.08 .08]); imagesc(reshape(U(7,:),imgsize)); title('Abundance #7','fontsize',8);axis off
    subplot_tight(5, 3, 11,[.08 .08]); imagesc(reshape(XT(7,:),imgsize)); title('GT #7','fontsize',8);axis off
    subplot_tight(5, 3, 12,[.08 .08]); imagesc(abs(reshape(XT(7,:),imgsize)-reshape(U(7,:),imgsize))); title('Diff #7','fontsize',8);axis off
   
    subplot_tight(5, 3, 13,[.08 .08]); imagesc(reshape(U(9,:),imgsize)); title('Abundance #9','fontsize',8);axis off
    subplot_tight(5, 3, 14,[.08 .08]); imagesc(reshape(XT(9,:),imgsize)); title('GT #9','fontsize',8);axis off
    subplot_tight(5, 3, 15,[.08 .08]); imagesc(abs(reshape(XT(9,:),imgsize)-reshape(U(9,:),imgsize))); title('Diff #9','fontsize',8);axis off
    drawnow;
    end
    fprintf('i = %d, obj = %.4f,res_p = %.4f, res_d = %.4f, mu = %.1f, SRE=%.2f, ||X-XT||=%.2f, [lambda=%.1e, beta=%.1e]\n',...
        i, obj1(i), res_p1(i), res_d1(i), mu,sre(i),xreds(i),lambda2,lambda1);

    i = i + 1;
end
if i == MaxIter + 1
    display('Maximum iteration reached!');
end
X = U;
end

function W = SSpasity(S,imgsize)
nr=imgsize(1);nc=imgsize(2);
[p,N]=size(S);
S=reshape(S',nr,nc,p);
W=zeros(p,1);
for i=1:p; 
    Saux=imfilter(S(:,:,i),fspecial('average',3));W(i,1)=1./norm(Saux,'fro');
end
W=repmat(W,1,nr*nc);
end

