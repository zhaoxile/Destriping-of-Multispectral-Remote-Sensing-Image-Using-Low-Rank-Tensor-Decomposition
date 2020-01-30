function [Tensor_X,Tensor_S] = LRTD_destriping(Tensor_Y,opts)
%% ==================================================================
%This demo is to restore the Multispectral (or Hyperspectral) image degraded by
%stripe noise using low rank tucker decomposition.
%--------------------------Objective function----------------------------
% min  1/2||Y-X-S||_F^2 + lambda_1||D_x(X)||_1 + lambda_2 ||D_z(X)||_1
%                              + lambda_3||S||_{2,1}
%                 s.t.     S = G*U1*U2*U3, U_i^T*U_i = I.
% where lambda_1, lambda_2, and lambda_3 are the regularizer parameters
% D_x and D_z are horizontal and spectral difference operator defined as follows
%-------D_x(X)(i,j,k) = X(i,j+1,k)-X(i,j,k)
%-------D_z(X)(i,j,k) = X(i,j,k+1)-X(i,j,k)
%% =================================================================== 
% INPUT:
%  Tensor_Y:     noisy 3-D image of size M*N*p normalized to [0,1]
%  opts:         The parameters involved in the model             
% OUTPUT:
%  Tensor_X:      3-D destriped image
%  Tensor_S:      estimated 3-D stripe component
%% ===================================================================
% Reference paper: 
% Yong Chen, Ting-Zhu Huang, Xi-Le Zhao, "Destriping of Multispectral
% Remote Sensing image Using Low-Rank Tensor Decomposition", IEEE Journal of 
% Selected Topics in Applied Earth Observations and Remote Sensing.
% -------------------------------------------------------------------------
%% Parameters
if isfield(opts,'maxit');          maxIter = opts.maxit;            else  maxIter = 1000;               end
if isfield(opts,'tol');            tol = opts.tol;                    else  tol = 1e-4;                  end
if isfield(opts,'lambda1') ;       lambda1 = opts.lambda1;       else lambda1 = 0.001;   end
if isfield(opts,'lambda2') ;       lambda2 = opts.lambda2;       else lambda2 = 0.001;   end
if isfield(opts,'lambda3') ;       lambda3 = opts.lambda3;       else lambda3 = 0.01;   end
if isfield(opts,'beta');           beta = opts.beta;                     else  beta = 0.1;               end
if isfield(opts,'rk');             rk = opts.rk;                     else  rk = [1 tsize(3) tsize(3)];               end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% init. variables 
tsize = size(Tensor_Y);
h    = tsize(1);
w    = tsize(2);
d    = tsize(3);
Tensor_X = Tensor_Y;
Tensor_S = zeros(tsize);
Tensor_W1 = zeros(tsize);
Tensor_W2 = zeros(tsize);
Tensor_W3 = zeros(tsize);
Tensor_Q1 = zeros(tsize(1),tsize(2)*tsize(3));
%%
Eny_y   = ( abs(psf2otf([+1, -1], [h,w,d])) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], [w,d,h])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
%% main loop
display  = 1;
isCont = 0;
[diffx_X,diffz_X] = diff3(Tensor_X,tsize); 
for iter = 1:maxIter
        fprintf('\n*****************************iter: %d ******************************\n', iter');
        
        Tensor_X_pre = Tensor_X; Tensor_S_pre = Tensor_S;
        %- Tensor_R subproblem
        Tensor_R1 = softThres( diffx_X - Tensor_W1/beta, lambda1/beta);
        Tensor_R2 = softThres( diffz_X - Tensor_W2/beta, lambda2/beta);
        
        %- Tensor_Q subproblem
        temp_Q = Tensor_S - Tensor_W3/beta;
        MatrixQ   = double(tenmat(tensor(temp_Q),1));
        for i = 1:size(MatrixQ,2)
        Tensor_Q1(:,i)=MatrixQ(:,i).*max(norm(MatrixQ(:,i))-lambda3/beta,0)/(norm(MatrixQ(:,i))+eps);
        end
        Tensor_Q = reshape(Tensor_Q1,tsize);
        %- S,G,U subproblem
        temp_x1 = ((Tensor_Y - Tensor_X) + beta*(Tensor_Q + Tensor_W3/beta))/(1+beta);
        Tensor_S  = double( tucker_als(tensor(temp_x1), rk, 'tol',1e-6,'printitn',0) );
        
        %Tensor_X subproblem
        [diffT_x,diffT_z] = diffT3(Tensor_R1 + Tensor_W1/beta, Tensor_R2 + Tensor_W2/beta,tsize);
        numer1 = (Tensor_Y - Tensor_S) + beta*diffT_x + beta*diffT_z;
        Tensor_X = real(ifftn(fftn(numer1)./(1 + beta*Eny_y + beta*Eny_z)));
        
        %- updating multipliers
        [diffx_X,diffz_X] = diff3(Tensor_X,tsize); 
        Tensor_W1 = Tensor_W1 + beta*(Tensor_R1 - diffx_X);
        Tensor_W2 = Tensor_W2 + beta*(Tensor_R2 - diffz_X);
        Tensor_W3 = Tensor_W3 + beta*(Tensor_Q - Tensor_S);
       %%
        relChgX = norm(Tensor_X(:) - Tensor_X_pre(:),'fro')/max(1,norm(Tensor_X_pre(:),'fro'));
        relChgS = norm(Tensor_S(:) - Tensor_S_pre(:),'fro')/max(1,norm(Tensor_S_pre(:),'fro'));
	    if  display
        fprintf('relChgX:%4.4e, relChgS: %4.4e\n', relChgX, relChgS);
        end
        if (iter> 40) &&  (relChgX < tol ) 
          disp(' !!!stopped by termination rule!!! ');  break;
        end
      %
      if  isCont
            nr1 = norm(Tensor_R1(:) - diffx_X(:), 'fro');
            nr2 = norm(Tensor_R2(:) - diffz_X(:), 'fro');
            nr3 = norm(Tensor_Q(:) - Tensor_S(:), 'fro');
            if display
               fprintf('nr1(R1-Dx): %4.4e, nr2(R2-Dz): %4.4e, nr3(Q-S): %4.4e\n',nr1,nr2,nr3);
            end
      end            
end
end
%%  subFunctions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the diff. of one 3-order tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [diff_x,diff_z] = diff3(tenX,sizeD)

dfx1     = diff(tenX, 1, 2);   % 水平方向
dfz1     = diff(tenX, 1, 3);   % 谱方向

dfx      = zeros(sizeD);
dfz      = zeros(sizeD);

dfx(:,1:end-1,:) = dfx1;
dfx(:,end,:)     = tenX(:,1,:) - tenX(:,end,:);
dfz(:,:,1:end-1) = dfz1;
dfz(:,:,end)     = tenX(:,:,1) - tenX(:,:,end);
diff_x = dfx;
diff_z = dfz;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  compute the inverse-diff. of one 3-order tensor 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [diffT_x,diffT_z] = diffT3(tenX, tenZ,sizeD)

dfx     = diff(tenX, 1, 2);
dfz     = diff(tenZ, 1, 3);

dfxT   = zeros(sizeD);
dfzT   = zeros(sizeD);

dfxT(:,1,:)     =  tenX(:,end,:) - tenX(:,1,:);
dfxT(:,2:end,:) = -dfx;
dfzT(:,:,1)     = tenZ(:,:,end) - tenZ(:,:,1);
dfzT(:,:,2:end) = -dfz;

diffT_x = dfxT;
diffT_z = dfzT;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  x^* = argmin_x 0.5*(x-a) + \lambda * |x|
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = softThres(a, tau)
x = sign(a).* max( abs(a) - tau, 0);
end


    
