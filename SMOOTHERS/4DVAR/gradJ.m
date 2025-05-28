% Calculate the gradient of J

function g = gradJ(xb,v,cv,inov,U,R_inv,dt,I,T,H,flag)

% Time between observation bins
if I > 1
  tau = (T-1)/(I-1);
else
  tau = 0;
end

g = cv + v;

% 4DVar gradient
if flag == 1

  MX_tl = M_tl(U*v,xb,dt,T);

  for t = 1:I
    tt    = tau*(t-1)+1;
    MX_ad = squeeze(H(:,:,t))'*R_inv*(squeeze(H(:,:,t))*MX_tl(:,tt) - inov(:,t));
    MX_ad = M_ad(MX_ad,xb,dt,tt-1);
    g = g + U'*MX_ad(:,1);
  end

% 4DEnVar gradient
elseif flag == 2

  for t = 1:I
    g = g + squeeze(U(:,:,t))'*R_inv*( squeeze(U(:,:,t))*v - inov(:,t) );
  end

end
