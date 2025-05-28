% Perform minimization of the hybrid cost function using conjugate gradient method.
% xg   <--- first guess (prior mean)
% xb   <--- background trajectory (updated between outer loops)
% xcv  <--- outer-loop perturbations
% xv   <--- inner-loop perturbations
% inov <--- innovations
% I    <--- number of observation time slots
% T    <--- assimilation window length
% U    <--- square root of error covariance
% HU   <--- measurement operator applied to square root of error covariance (4denvar only)
% H    <--- observation operator
% R_i  <--- inverse of observervation error covariance matrix
% flag <--- set to 1 for 4DVar with adjoint and 2 for 4DVar without adjoint

function [xcv,cv,J] = cg_minimize(xg,xb,cv,inov,dt,I,T,R_i,U,H,ntmax,flag,HU)

% Time between observation bins
if I > 1
  tau = (T-1)/(I-1);
else
  tau = 0;
end

% Plot first guess
%figure(3+flag+1); hold off
%xd = [0:length(xg)-1];
%plot(xd,xb(1,:)','b','linewidth',2); hold on;

% Set inner-loop control variable and x to zero
v   = cv*0;
xv  = xg*0;

% Set initial d and r to be negative the gradient
if flag == 1
  d = -gradJ(xb,v,cv,inov,U,R_i,dt,I,T,H,flag);
else
  d = -gradJ(xb,v,cv,inov,HU,R_i,dt,I,T,H,flag);
end
r = d;

gdot0   = d'*d;
gdot1   = gdot0;
norm(1) = sqrt(gdot0);

%disp(' ')
%disp(['Target gradient norm will be ',num2str(norm(1)*1e-4)])
%disp(' ')

for i=1:ntmax+1

  % Calculate Jb
  Jb = (cv+v)'*(cv+v)/2;

  % Calculate Jo
  Jo = 0;
  if flag == 1 % 4DVar
    MX_tl = M_tl(xv,xb,dt,T);
    for t = 1:I
      tt = tau*(t-1)+1;
      Jo = Jo + ( squeeze(H(:,:,t))*MX_tl(:,tt) - inov(:,t) )'*R_i*( squeeze(H(:,:,t))*MX_tl(:,tt) - inov(:,t) );
    end
  elseif flag == 2 % 4DEnVar
    for t = 1:I
      Jo   = Jo   + ( squeeze(HU(:,:,t))*v - inov(:,t) )'*R_i*( squeeze(HU(:,:,t))*v - inov(:,t) );
    end
  end
  Jo = Jo/2;

  norm(i+1) = sqrt(gdot1);
%  disp([])
%  disp(['Iteration: ',num2str(i-1),'  Grad J: ',num2str(norm(i+1)),'  Jb: ',num2str(Jb),'  Jo: ',num2str(Jo)])

  J(i) = Jb + Jo;

  % Transform back to model space
  xcv = U*cv;
  xv  = U*v;

%  figure(3+flag+1)
%  plot(xd,xcv+xv+xg,'r'); hold on;

  if norm(i+1)<norm(1)*1e-8; break; end

  % Calculate A*d
  dum1 = cv*0;
  dum2 = inov*0;
  if flag == 1
    Ad = gradJ(xb,d,dum1,dum2,U,R_i,dt,I,T,H,flag);
  else
    Ad = gradJ(xb,d,dum1,dum2,HU,R_i,dt,I,T,H,flag);
  end

  % Update CG variables
  a = gdot1/(d'*Ad);

% TEMP: REDUCE STEP SIZE FOR NONLINEAR MEAURMENT OPERATOR
if flag == 1
%  a = 0.6*a;
end

  r = r - a*Ad;
  v = v + a*d;
  gdot0 = gdot1;
  gdot1 = r'*r;
  bet   = gdot1/gdot0;
  d     = r + bet*d;

end  

% Add inner-loop perturbations to outer-loop perturbations
xcv = xcv + xv;
cv  = cv  + v;

%figure(3+flag+1)
%plot(xd,xcv+xg,'r','linewidth',2); hold on;
