% Function for performing the particle filter update step
%
% * This version performs tempering steps based on regularization coefficients
%
%  INPUT 
%           x: prior particles (Nx x Ne)
%          hx: obs-space prior particles (Ny x Ne)
%           y: observation vector  (Ny x 1)
%       var_y: obs error variance
%
%  OUTPUT 
%        xmpf: posterior mean      (Nx x 1)
%          xa: posterior particles (Nx x Ne)

function [xmpf,xa] = pf_update_bootstrap(x,hx,y,var_y,qcpass)

% Get array dimensions
[Nx,Ne] = size(x);
Ny = length(y);

% Nothing to do if no obs pass QC
xmpf = mean(x')';
e_flag = 0;
if sum(qcpass) == length(y)
  return
end

% Remove obs that don't pass QC
ind = find(qcpass==1);
y(ind) = [];
hx(ind,:) = [];
Ny = length(y);

% Innovations
innov = 0;
xa = x;
hxa = hx;
for i = 1:Ny % Observation loop

  % Normalized squared innovations
  d = (y(i) - hx(i,:)).^2./(2.*var_y);
  pf_infl = max(1,max(d)/300);
  innov = innov + d ./ pf_infl;

  d = (y(i) - hxa(i,:)).^2./(2.*var_y);
%  pf_infl = max(1,max(d)/300);
  sinnov = d ./ pf_infl / 2;

  % Weights for moment estimation
  wo = exp( -innov );
  wo = wo./sum(wo);

  % Weights for sampling
  w = exp( -sinnov );
  w = w./sum(w);

  % Posterior moments
  xmpf = 0; hxmpf = 0;
  for n = 1:Ne
    xmpf = xmpf + wo(n)*x(:,n);
    hxmpf = hxmpf + wo(n)*hx(:,n);
  end
  avar = 0; havar = 0;
  for n = 1:Ne
    avar = avar + wo(n)*(x(:,n)-xmpf).^2;
    havar = havar + wo(n)*(hx(:,n)-hxmpf).^2;
  end
  avar = avar * Ne / (Ne-1);
  havar = havar * Ne / (Ne-1);

  % Apply deterministic resampling
  ind = sampling(hxa(i,:),w,Ne);
  xa = xa(:,ind);
  hxa = hxa(:,ind);

  % Adjust mean and variance
  smea = mean(xa')'; svar = var(xa')';
  shmea = mean(hxa')'; shvar = var(hxa')';  
  for n = 1:Ne
%    xa(:,n) = xmpf + (xa(:,n) - smea) .* sqrt( avar./svar );
%    hxa(:,n) = hxmpf + (hxa(:,n) - shmea) .* sqrt( havar./shvar );
%    xa(:,n) = xmpf + (xa(:,n) - smea);
%    hxa(:,n) = hxmpf + (hxa(:,n) - shmea);
  end

end

% Weights
%w = exp( -innov );
%w = w./sum(w);

%% Apply deterministic resampling
ind = sampling(1:Ne,wo,Ne);
xa = x(:,ind);

xmpf = mean(xa')';
pvar = var(x')';
pvar = reshape(pvar(1:Nx/2),sqrt(Nx/2),sqrt(Nx/2));
avar = var(xa')';

% TEMP: for rankine vortex
if 1 == 0
  fig = figure(30); 
  h = subplot(1,3,3);
  pname = 'Bootstrap PF Prior/Posterior \sigma_x'; plet = 'c)';

  avar = reshape(avar(1:Nx/2),sqrt(Nx/2),sqrt(Nx/2));
  ticks = [0.1:0.1:2];
  dum = sqrt(avar./pvar)';
  contourf(dum,ticks); hold on
  set(gca,'clim',[0.1,2],'ColorScale','log');
  cb = colorbar();
  cb.Ruler.Scale = 'log';
  cb.Ruler.MinorTick = 'on';
  dum(dum<1) = 0.99;
  contour(dum,ticks,'linewidth',3,'color','r');
  
  axis square
  set(gca,'xtick',[20:20:100],'ytick',[20:20:100])
  xlabel('Grid point','fontsize',14);
  title(pname,'fontsize',18,'FontWeight','Normal');
  text(0.02,0.95,plet,'fontsize',16,'units','normalized')
  
  
  io = 40; jo = 52;
  text(io,jo,'*','fontsize',16)
  
  p = get(h,'pos');
  p(3) = 0.22;
  set(h,'pos',p);
  
  % Save figure
  fpath = '/wave/Users/poterjoy/PF_PLAYGROUND/TESTING/FIGS/spread.pdf';
  set(fig,'PaperSize',[12,4])
  set(fig,'PaperPositionMode','manual'); 
  set(fig,'PaperPosition',[0,0,12,4]);
  disp(['Saving file to ',fpath])
  saveas(fig,fpath,'pdf');
  
end
