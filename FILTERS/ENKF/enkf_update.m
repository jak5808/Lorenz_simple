% Function for performing enkf update using Whitaker and Hamill filter with
% Anderson adaptive state-space inflation
%
%  INPUT 
%           x: prior ensemble (Nx x Ne)
%          hx: obs-space prior ensemble (Ny x Ne)
%           y: observation vector  (Ny x 1)
%       var_y: obs error variance
%          HC: matrix determining localization between obs- and model-space  (Ny x Nx)
%         HCH: matrix determining localization between obs- and obs-space  (Ny x Ny)
%    inf_flag: flag for inflation option
%         inf: prior mean of inf coefficients (Nx x 1)
%     var_inf: prior variance of inf coefficients
%       gamma: relaxation parameter for RTPS
%
%  OUTPUT 
%          xm: posterior mean (Nx x 1)
%           x: posterior ensemble (Nx x Ne)
%         inf: posterior mean of inf coefficients (Nx x 1)
%      e_flag: error flag

function [xm,x,e_flag,prior_inf,prior_inf_y] = enkf_update(x,hx,y,var_y,HC,HCH,inf_flag,prior_inf,prior_inf_y,var_inf,gamma,qcpass)

% Nothing to do if no obs pass QC
xm = mean(x')';
e_flag = 0;
if sum(qcpass) == length(y)
  return
end

% Get array dimensions
[Nx,Ne] = size(x);
Ny = length(y);

% Perform prior inflation
if inf_flag == 1
  [xm,xp,prior_inf] = inflation_anderson(x,hx,y,prior_inf,var_inf,var_y,HC,0.8,qcpass);
  [hxm,hxp,prior_inf_y] = inflation_anderson(hx,hx,y,prior_inf_y,var_inf,var_y,HCH,0.8,qcpass);
else
  xm = mean(x')';
  xp = x - xm;
  hxm = mean(hx')';
  hxp = hx - hxm;
end

% Save original perturbations for RTPP
xpo = xp;

% Observation loop
for i = 1:Ny

  % QC check
  if qcpass(i) > 0, continue, end

  % Innovations
  d = y(i) - hxm(i);
  hxo = hxp(i,:);
  var_den = hxo*hxo'/(Ne-1) + var_y;

  % --- State-space update ---

  % Calculate localized gain matrix
  P = xp*hxo'/(Ne-1);
  P = P.*HC(i,:)';
  K = P/var_den;

  % Update mean
  xm = xm + K*d';

  % Update perturbations 
  beta = 1/(1 + sqrt( var_y/var_den ));
  xp = xp - beta*K*hxo;

  % Error check
  if sum(sum(isnan(xm))) > 0
    e_flag = 1; 
    x = xp;
    return
  else
    e_flag = 0;
  end

  % --- Obs-space update ---

  % Calculate localized gain matrix
  P = hxp*hxo'/(Ne-1);
  P = P.*HCH(i,:)';
  K = P/var_den;

  % Update mean
  hxm = hxm + K*d';

  % Update perturbations 
  beta = 1/(1 + sqrt( var_y/var_den) );
  hxp = hxp - beta*K*hxo;

end

% TEMP: for rankine vortex
if 1 == 0
fig = figure(30); 
h = subplot(1,3,1);
pname = 'EnKF Prior/Posterior \sigma_x'; plet = 'a)';

pvar = var(xpo')';
pvar = reshape(pvar(1:Nx/2),sqrt(Nx/2),sqrt(Nx/2));

avar = var(xp')';
avar = reshape(avar(1:Nx/2),sqrt(Nx/2),sqrt(Nx/2));
ticks = [0.1:0.1:2];
dum = sqrt(avar./pvar)';
contourf(dum,ticks); hold on;
set(gca,'clim',[0.1,2],'ColorScale','log');
%cb = colorbar();
%cb.Ruler.Scale = 'log';
%cb.Ruler.MinorTick = 'on';
dum(dum<1) = 0.99;
contour(dum,ticks,'linewidth',3,'color','r');

axis square
set(gca,'xtick',[20:20:100],'ytick',[20:20:100])
xlabel('Grid point','fontsize',14);
ylabel('Grid point','fontsize',14);
title(pname,'fontsize',18,'FontWeight','Normal');
text(0.02,0.95,plet,'fontsize',16,'units','normalized')

io = 40; jo = 52;
text(io,jo,'*','fontsize',16)


p = get(h,'pos');
p(3) = 0.22;
set(h,'pos',p);

pause(5)
end

% Apply relaxation-based inflation
switch inf_flag
  case 2 % Apply RTPS
    v1 = sqrt(var(xpo')');
    v2 = sqrt(var(xp')');
    xp = xp.*( gamma*(v1 - v2)./v2 + 1 );
  case 3 % Apply RTPP
    xp = xp*(1-gamma) + xpo*gamma;
end


% Add perturbations to mean
x = xm + xp;

if sum(isnan(xm)) > 0
  xm = nan;
  x = nan
end
