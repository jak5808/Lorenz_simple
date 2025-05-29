% ==========================================================
%
%  Main script for running idealized data assimilation experiments
%  with Lorenz models.
%
%  The current version supports three models: Lorenz (1963),
%  Lorenz (1996), and Lorenz (2005)
%
% ==========================================================
%
% The default model is Model III of Lorenz (2005)
%   https://doi.org/10.1175/JAS3430.1
%
% The EnKF is based on Whitaker and Hamill (2002):
%   https://doi.org/10.1175/1520-0493(2002)130<1913:EDAWPO>2.0.CO;2
%
% The local PF is based on Poterjoy (2022):
%   https://doi.org/10.1002/qj.4328
%

% Link matlab libraries needed for experiments
da_libs

close all; clear all; warning off

%  ........................................................
% Choose a model with m_flag:   1 <== Lorenz 63
%				                2 <== Lorenz 96
%				                3 <== Lorenz 05 (model III)
m_flag = 3;
%
%  ........................................................
%  Specify filters and smoothers with f_flag{} and s_flag{}
%  ........................................................
%  :      f_flag options       :      s_flag options      :
%  ........................................................
%  : 1 <== EnKF		       : 1 <== E4DVar		     
%  : 2 <== Local PF	       : 2 <== 4DEnVar

f_flag{1} = 1;   s_flag{1} = 0;
%f_flag{2} = 2;	 s_flag{2} = 0;

% Experiment names
e_name{1} = 'EnKF'; 
%e_name{2} = 'LPF';

%  --------------------------------------------------------
% | The first part of the code contains all the experiment |
% | parameters, such as model and observation information  |
% | and filter parameters.                                 |
%  --------------------------------------------------------

% --- Model parameters ---
T   = 100;  % number of obs times
I   = 1;    % number of obs times between smoother steps (Var only)
            % (I > 1 results in smoother)
Ne  = 80;   % number of members

% Fix T so it is divisible by I
T = T + mod(T,I);

% Experiment flags
plot_flag = 0;  % Set to 1 to plot prior values each filter time
disp_flag = 1;  % Set to 1 to display mean RMSEs each filter time
h_flag    = 0;  % Set to 0 for H(x) = x
                % Set to 1 for H(x) = x^2
                % Set to 2 for H(x) = log(|x|)

% --- Observation parameters ---
sig_y = 0.1;  % observation error standard deviation
tau   = 1;    % model time steps between observation times
obf   = 4;    % observation spatial frequency: spacing between variables
obb   = 0;    % observation buffer: number of variables to skip when generating obs
              % - setting a non-zero obb will create a data void in the domain

% --- EnKF parameters ---
roi_kf   = 0.005; % EnKF localization radius of influence
gamma    = 0.7;   % RTPS/RTPP parameter (options 2 or 3)
inf_flag = 2;     % 0 ==> no inflation
		          % 1 ==> State-space adaptive inflation (Anderson)
                  % 2 ==> Relaxation to prior spread (RTPS)
    	          % 3 ==> Relaxation to prior perturbations (RTPP)

% See: https://doi.org/10.1175/MWR-D-11-00276.1

% --- PF parameters ---
roi_pf = 0.001;     % PF localization radius of influence
alpha = 0.3;        % PF mixing coeffiecient
Neff =  0.80*Ne;    % effective ensemble size used for regularization
min_res = 0.0;      % minimum residual left after tempering with regularization
pf_kddm = 0;        % PF KDDM option

% --- 4DVar parameters ---
roi_var = 0.005;    % Var localization radius of influence
Ni = 200;           % max number of inner iterations
No = 5;             % number of outer iterations

% --- Hybrid parameter ---
beta = 0.0;  % weighting applied to static error covariance
             % ensemble error covariance is weighted by (1-beta) 

% Number of filter options
Nf = length(f_flag);

% Model parameters are determined by case block for selected model 
switch m_flag

  % Lorenz 63 model parameters
  case 1
    Nx  = 3;    % number of variables
    dt  = 0.01; % time step
    s = 10; 
    r = 28; 
    b = 8/3;

  % Lorenz 96 model parameters
  case 2, 
    Nx  = 40;   % number of variables
    dt  = 0.05; % time step
    F   = 8;    % forcing term (truth)
    Fe  = 8;    % forcing term (model)

    % The TL and AD model for 4DVar are based on a centered-
    % in-time finite differencing scheme. This version requires 
    % a smaller time step than the dt = 0.05 used in most studies.
    % To keep things consistent with other work, tau and dt 
    % are specified above with an assumed dt = 0.05 time step
    % and adjusted here.
    if dt == 0.05 && I > 1
      dt = 0.01;
      tau = 5*tau;
    end

    % Load climatological prior error covariance
    load('MODELS/be_l96.mat')

    % Estimate square root of B
    [U S V] = svd(B);
    S(S<0)=0;
    Bsqrt = U * sqrt(S) * V';
    Bsqrt(Bsqrt<0) = 0;

  % Two-scale Lorenz 05 model parameters
  case 3, 

    Nx = 960;   % model variables
    dt = 0.05;  % time step
    F  = 15;    % forcing term (truth)
    Fe = 15;    % forcing term (model)
    K  = 32;    % spatial smoothing parameter
    Im = 12;
    b  = 10.0;
    c  = 2.5;

end

% Set ob and model bias
obbias = 0
modbias = 0.5

% Start parallel run
delete(gcp('nocreate'))
poolobj = parpool(8);

% Use same random numbers each experiment
rng(1); 

%  --------------------------------------------------------
% | The next part of the code sets up the experiment. It   |
% | generates the measurement operator, localization       |
% | matrix, and obs error covariance matrix, spins up the  |
% | model, creates a truth simulation and observations,    |
% | and generates the initial prior ensemble. The truth    |
% | simulation is formed by running the model 1000 time    |
% | steps + the length of the experiment specified with T. |
% | The first prior ensemble comes from an ensemble        | 
% | forecast initialized from the randomly perturbed truth |
% | state.                                                 |
%  --------------------------------------------------------

% Generate observation error covariance matrix
var_y  = sig_y^2; 

numobs = ceil((Nx-2*obb)/obf);
R      = eye(numobs)*var_y;
R_i    = inv(R);

% Define H
H  = eye(Nx);
H  = H(obb+1:obf:Nx-obb,:);
Ny = length(H(:,1));

% Correlation matrix for localization
C_kf  = gen_be_periodic(roi_kf,1,Nx,1/Nx);
C_pf  = gen_be_periodic(roi_pf,1,Nx,1/Nx);
C_var  = gen_be_periodic(roi_var,1,Nx,1/Nx);

% Apply interpolation part of measurement operator 
C_kf = H*C_kf;
C_pf = H*C_pf;

% Take square root for Var formulation
[U S V] = svd(C_var); 
S(S<0)=0;
Csqrt_var = U * sqrt(S) * V';

% Take inverse of Csqrt_var for var step
C_inv = inv(Csqrt_var/sqrt(Ne-1));

% Define HV (for verification)
ind=1:Nx;
ind(obb+1:obf:Nx-obb) = [];
HV = eye(Nx);

% Define domain
xd = [1:Nx]';

% Initialize model for spinup period
xt(1:Nx,1) = 3*sin([1:Nx]/(6*2*pi));

% Spin up initial truth state
switch m_flag
  case 1
    xt = M_nl_l63(xt,dt,1000,s,r,b);
  case 2
    xt = M_nl_l96(xt,dt,1000,F);
  case 3
    xt = M_nl_l05III(xt,dt,1000*0.05/dt,K,Im,b,c,F);
end

% Run initial ensemble forecast
parfor n = 1:Ne
%for n = 1:Ne

  dum = xt + 1*randn(Nx,1);
  switch m_flag
    case 1
      xi(:,n) = M_nl_l63(dum,dt,100,s,r,b);
    case 2
      xi(:,n) = M_nl_l96(dum,dt,100,Fe);
    case 3
      xi(:,n) = M_nl_l05III(dum,dt,100,K,Im,b,c,F);
  end

end

% Generate Truth
switch m_flag

  case 1
    xt = M_nl_l63(xt,dt,100,s,r,b);
    for t = 2:T
      xt(:,t) = M_nl_l63(xt(:,t-1),dt,tau,s,r,b);
    end

  case 2
    xt = M_nl_l96(xt,dt,100,F);
    for t = 2:T
      xt(:,t) = M_nl_l96(xt(:,t-1),dt,tau,F);
    end

  case 3
    xt = M_nl_l05III(xt,dt,100,K,Im,b,c,F);
    for t = 2:T
      xt(:,t) = M_nl_l05III(xt(:,t-1),dt,tau,K,Im,b,c,F);
    end

end

% Create synthetic obs from truth and add random errors
dum = randn(T,Nx)'*sig_y;

switch h_flag
  case 0 % ---   H(x) = x + eps   ---
    Y = H*( xt + dum );
  case 1 % ---   H(x) = x^2 + eps   ---
    Y = H*( (xt.^2 + dum ) );  
  case 2 % ---   H(x) = log(abs(x)) + eps   ---
    Y = H*( log(abs(xt + dum )) );
end

% Include obs bias
Y = Y + obbias;

% Initialize prior ensemble and deterministic state for all experiments
for f = 1:Nf
  x{f} = xi;
  if s_flag{f} > 0
    xdet{f} = mean(xi')';
  end
end
clear xi

for f = 1:Nf
  % Initialize inflation values for adaptive inflation
  prior_inf{f} = ones(Nx,1);
  prior_inf_y{f} = ones(Ny,1);
  var_inf = 0.8;

  % Preallocate arrays for output statistics % Knisely
  prior_mean{f} = zeros(Nx,T);
  post_mean{f} = zeros(Nx,T);
  prior_spread{f} = zeros(Nx,T);
  post_spread{f} = zeros(Nx,T);
  innov_mean{f} = zeros(Ny,T);
  anal_incr{f} = zeros(Nx,T);
end

%  --------------------------------------------------------
% | This part of the code loops through the observation    |
% | times. It applies the local PF to update the ensemble  |
% | then runs an ensemble forecast from updated particles. |
%  --------------------------------------------------------

e_flag = 0;
for t = 1:T % Time loop

  % Add model bias to prior
  for f = 1:Nf
    x{f} = x{f} + modbias;
  end

  % Plot prior information for each filter time
  if plot_flag

    for f = 1:Nf
      figure(f)
      % Plot prior particles 
      subplot(3,1,1); hold off
      for n = 1:Ne
        plot(x{f}(:,n),'color','b','linewidth',2); hold on;
      end      

      title(['Prior particles (',e_name{f},')'],'fontsize',20)

      % Plot observations
      scatter(H*xd,Y(:,t),'k','linewidth',2);

      % Plot truth
      plot(xt(:,t),'color','g','linewidth',2);
      xlim([0,Nx+1]);

      % Plot histogram for variable 1
      subplot(3,1,2); hold off;
      hist(x{f}(1,:),Nx)
      title('X1 prior histogram','fontsize',20)
    end

  end

  % Loop through experiments and perform update based on specified options
  for f = 1:Nf

    % --------------------------
    % ----- Smoother Step ------
    % --------------------------

    % Index of obs in current obs window used by smoothers
    i_obs = mod(t-1,I)+1;

    % When a smoother is specified, obs are assimilated once they 
    % are all collected over window.
    switch s_flag{f}

      % 4DVar with tangent linear and adjoint
      case 1

        % Perform update for all obs over window 
        if i_obs == 1

          % Store deterministic solution and generate square root of hybrid error
          % covariance for E4DVar at start of each obs window; see Buehner (2005)
          % for description of this step.

          xmea = mean(x{f}')';
          Q{f} = [];
          for n = 1:Ne
      	    xp = x{f}(:,n) - xmea;
            Q{f} = [Q{f},diag(xp)*Csqrt_var/sqrt(Ne-1)];
          end
  
          % Modify Q to include weighted square root of B in first block
          % and weighted localized ensemble square root in remaining portion.
          % This results in a square root of the localized hybrid error covariance.
          Q{f} = [sqrt(beta)*Bsqrt,sqrt(1-beta)*Q{f}];

          % Set up first guess state, background, and control variables
          xg(:,1) = xdet{f};
          %xg(:,1) = xmea; % Option to use mean instead of deterministic
          xb = xg;
          cv = zeros(Nx*(Ne+1),1);

          % Get full nonlinear trajectory for TL and AD models
          % and calculate innovations
          k = 0;
          HTL = []; R_s = R_i;

          for i = 1:(I-1)*tau+1
            if i > 1
              xb(:,i) = M_nl_l96(xb(:,i-1),dt,1,Fe);
            end
            if mod(i-1,tau) == 0
              k = k + 1;
              switch h_flag
                case 0
                  hx = H*xb(:,i);
                  HTL(:,:,k) = H;
                case 1
                  hx = H*( (xb(:,i).^2 ) );
                  HTL(:,:,k) = H*diag(2*xb(:,i));
                case 2
                  hx = H*log(abs(xb(:,i)));
                  HTL(:,:,k) = H*diag(1./xb(:,i));
              end
      	      d(:,k) = Y(:,t+k-1) - hx;

            end
          end

          % Outer-loop iterations to minimize 4DVar cost function
          for il = 1:No

            % Minimize cost function using CG
            [xu,cv,J] = cg_minimize(xg,xb,cv,d,dt,I,(I-1)*tau+1,...
		       R_i,Q{f},HTL,Ni,1);

            % Plot J
            %figure(20+f); hold off;
            %semilogy(J,'b','linewidth',2);

            % Update trajectory and innovations
            k = 0;
            HTL = [];
            for i = 1:(I-1)*tau+1
  
              if i == 1
                xb(:,i) = xg + xu;
              else
                xb(:,i) = M_nl_l96(xb(:,i-1),dt,1,Fe);
              end
  
              % Update innovations if using additional outer loops
              if mod(i-1,tau) == 0
                k = k + 1;
                if il < No
                  switch h_flag
                    case 0
                      hx = H*xb(:,i);
                      HTL(:,:,k) = H;
                    case 1
                      hx = H*( xb(:,i).^2 );
                      HTL(:,:,k) = H*diag(2*xb(:,i));
                    case 2
                      hx = H*log(abs(xb(:,i)));
                      HTL(:,:,k) = H*diag(1./xb(:,i));
                    end
           	    d(:,k) = Y(:,t+k-1) - hx;
                end

              end % Innovations
  
            end  % Trajectory

          end % Outer loops

          % Store deterministic analysis
          xdet{f} = xb(:,1);

          % Clear variables for next cycle
          clear d xb

        end % End assimilation

      % 4DVar ensemble approximation without tangent linear and adjoint (4DEnVar)
      case 2

        % Perform update for all obs over window
        if i_obs == 1

          xmea = mean(x{f}')';
          Q{f} = [];
          for n = 1:Ne
      	    xp = x{f}(:,n) - xmea;
            if f_flag{f} == 2
              Q{f} = [Q{f},diag(xp)*Csqrt_var./sqrt(Ne-1)];
            else
              Q{f} = [Q{f},diag(xp)*Csqrt_var./sqrt(Ne-1)];
            end
          end
  
          % Modify Q to include weighted square root of B in first block
          % and weighted localized ensemble square root in remaining portion.
          % This results in a square root of the localized hybrid error covariance.
          Q{f} = [sqrt(beta)*Bsqrt,sqrt(1-beta)*Q{f}];

          % Innovations and perturbations for obs-space Q matrix
          k = 0;
          HQ{f} = [];
          for i = 1:(I-1)*tau+1

            % Ensemble trajectory 
            for n = 1:Ne
              if i == 1
                xb(:,n) = x{f}(:,n);
              else
                xb(:,n) = M_nl_l96(xb(:,n),dt,1,Fe);
              end
            end

            % Deterministic trajectory 
            if i == 1
              xg = xdet{f};
            else
              xg = M_nl_l96(xg,dt,1,Fe);
            end

            % Calculate obs-space prior and innovations at obs times
            if mod(i-1,tau) == 0
  
              k = k + 1;
              for n = 1:Ne
                switch h_flag
                  case 0
                    hxn(:,n) = H*xb(:,n);
                  case 1
                    hxn(:,n) = H*((xb(:,n).^2));
                  case 2
                    hxn(:,n) = H*log(abs(xb(:,n)));
                end
              end

              switch h_flag
                case 0
                  hxm = H*xg;
                case 1
                  hxm = H*((xg.^2));
                case 2
                  hxm = H*log(abs(xg));
              end

              % Innovations
              xmea = mean(hxn')';
    	      d(:,k) = Y(:,t+k-1) - hxm;

              % Obs-space prior perturbations for pre-conditioning matrix
              Qt = [];
              for n = 1:Ne
                hxn(:,n) = hxn(:,n) - xmea;
                Qt = [Qt,diag(hxn(:,n))*H*(Csqrt_var')./sqrt(Ne-1)];
              end

              % Modify HQ to include weighted square root of B in first block
              % and weighted localized ensemble square root in remaining portion.
              % This results in a square root of the localized hybrid error covariance.
              HQ{f}(:,:,k) = [sqrt(beta)*H*Bsqrt,sqrt(1-beta)*Qt];

            end % Obs times
    
          end % Loop through times in obs window

          % Set up first guess state and control variables
          xb = xdet{f};
          xg = xb;
          cv = zeros(Nx*(Ne+1),1);

          % Minimize cost function using CG
          [xu,cv,J] = cg_minimize(xg,xb,cv,d,dt,I,(I-1)*tau+1,...
 				    R_i,Q{f},H,Ni,2,HQ{f});

          % Plot J
          %figure(30); hold off;
          %semilogy(J,'b','linewidth',2);

          % Store deterministic analysis
          xdet{f} = xg + xu;

          % Clear variables for next cycle
          clear d xb hxn

        end % End assimilation

    end

    % --------------------------
    % ------ Filter Step -------
    % --------------------------

    % Obs-space priors
    switch h_flag
      case 0; hx = H*x{f};
      case 1; hx = H*( (x{f}.^2 ) );
      case 2; hx = H*log(abs(x{f}));
    end

    % QC step
    qcpass = zeros(1,Ny);
    for i = 1:Ny
      d = abs(Y(i,t) - mean(hx(i,:)));
      if d > 4 * sqrt( var(hx(i,:)) + var_y  )
        qcpass(i) = 1;
      end
    end
    clear d

    % TEMP: turn off
    qcpass = zeros(1,Ny);
    
    % Save prior ens mean, ens spread, and innovation % Knisely 
    x_prior_mean = mean(x{f}, 2);
    prior_mean{f}(:,t) = x_prior_mean;

    for i = 1:Nx
      prior_spread{f}(i,t) = std(x{f}(i,:));
    end

    hx_mean = mean(hx, 2);
    innov_mean{f}(:,t) = Y(:,t) - hx_mean;

    % Call filter 
    switch f_flag{f}

      case 1 % EnKF update step

        [xm{f},x{f},e_flag,prior_inf{f},prior_inf_y{f}] = enkf_update(x{f},hx,Y(:,t),...
            var_y,C_kf,C_kf*H',inf_flag,prior_inf{f},prior_inf_y{f},var_inf,gamma,qcpass);

      case 2 % Local PF update step

        [xm{f},x{f},e_flag] = pf_update(x{f},hx,Y(:,t),C_pf,C_pf*H',Neff,min_res,alpha,var_y,pf_kddm,qcpass,10);

      case 3 % Bootstrap PF update step

        [xm{f},x{f}] = pf_update_bootstrap(x{f},hx,Y(:,t),var_y);

        % Add noise to increase particle diversity
        x{f} = x{f} + randn(Nx,Ne)*0.1;

    end

    % Set all values to nan if filter fails
    if max(e_flag) == 1
      xm{f} = xm{f}*nan;
      x{f} = x{f}*nan;
    end

    % Center ensemble on var analysis
    if s_flag{f} > 0
      if mod(t,I) == 0
        for n = 1:Ne
          x{f}(:,n) = x{f}(:,n) - xm{f} + xdet{f};
        end
      end
    end

    % Save post ens mean, ens spread, and analysis increment % Knisely 
    post_mean{f}(:,t) = xm{f};
    
    for i = 1:Nx
      post_spread{f}(i,t) = std(x{f}(i,:));
    end
    
    anal_incr{f}(:,t) = xm{f} - x_prior_mean;
    
    % Save solution for verification
    if (s_flag{f} > 0) && (mod(t,I) == 0)
      xm{f} = xdet{f};
    end

  end  % DA experiment loop

  % Plot posterior information for each filter time
  if plot_flag

    for f = 1:Nf

      figure(f)
      subplot(3,1,3); hold off
      for n = 1:Ne
        plot(x{f}(:,n),'color','b','linewidth',2); hold on;
        title('PF posterior','fontsize',20)
        xlim([0,Nx+1]);
      end
      scatter(H*xd,Y(:,t),'k','linewidth',2);
      plot(xt(:,t),'color','g','linewidth',2);
      plot(xm{f},'color','r','linestyle','--','linewidth',2);
      pause(1)

    end
  end

%  --------------------------------------------------------
% | The next part of code in the time loop calculates and  |
% | saves some basic statistics from the experiments.      |
%  --------------------------------------------------------

  % Keep track of posterior errors during experiments
  if mod(t,I) == 0

    % Number of verification times
    tt = t/I;

    if disp_flag
      tstep = sprintf('%2.2f',t);
      textl = ['Time = ',tstep];
    end

    for f = 1:Nf

      % RMSE errors
      dif = HV*xt(:,t) - HV*xm{f}; 
      err{f}(tt) = sqrt(mean(dif.*dif));

      % Spread
      xvar = 0;
      for n = 1:Ne
        xvar = xvar + (HV*(x{f}(:,n) - xm{f})).*(HV*(x{f}(:,n) - xm{f}))./(Ne-1);
      end

      sig{f}(tt) = sqrt(mean(xvar));

      % Values for display
      if disp_flag
        rmse = sprintf('%2.4f',err{f}(tt));
        textl = [textl,', RMSE (',e_name{f},') = ',rmse];
      end

    end

    % Print errors and spread at each time
    if disp_flag
      disp(textl)
    end

  end

  % Run ensemble forecast for next cycle
  for f = 1:Nf
    dum = x{f};
    parfor n = 1:Ne
%    for n = 1:Ne
      switch m_flag
        case 1 % Lorenz 63
          dum(:,n) = M_nl_l63(dum(:,n),dt,tau,s,r,b);
        case 2 % Lorenz 96
          dum(:,n) = M_nl_l96(dum(:,n),dt,tau,Fe);
        case 3 % Lorenz 05
          dum(:,n) = M_nl_l05III(dum(:,n),dt,tau,K,Im,b,c,F);
      end
    end
    x{f} = dum;
  end

  % Run deterministic forecast to observation time (Var cases only)
  for f = 1:Nf
    if s_flag{f} > 0
      switch m_flag
        case 1 % Lorenz 63
          xdet{f} = M_nl_l63(xdet{f},dt,tau,s,r,b);
        case 2 % Lorenz 96
          xdet{f} = M_nl_l96(xdet{f},dt,tau,Fe);
        case 3 % Two-scale Lorenz 05
          xdet{f} = M_nl_l05III(xdet{f},dt,tau,K,Im,b,c,F);
      end
    end
  end

end % Time loop

%  --------------------------------------------------------
% | The last part makes figures and saves data from each   |
% | experiment.                                            |
%  --------------------------------------------------------

fig = figure(2); hold off;

cl = colormap(lines(Nf));
for f = 1:Nf
  plot(err{f},'color',cl(f,:),'linewidth',2,'linestyle','-'); hold on;
  plot(sig{f},'color',cl(f,:),'linewidth',2,'linestyle','--');

  % Take average of rmse and spread
  mea(f)  = mean(err{f}(100:end));
  sigd(f) = mean(sig{f}(100:end));

  % Legend
  leglab{2*f-1} = [e_name{f},' RMSE (average: ',num2str(mea(f)),')'];
  leglab{2*f} = [e_name{f},' spread (average: ',num2str(sigd(f)),')'];

end

xlim([0,tt+1]);
ylabel('Mean RMSE/spread','fontsize',20)
xlabel('Cycle number','fontsize',20)
set(gca,'fontsize',16);

lh = legend(leglab,'location','northwest');
set(lh,'box','off')

disp(' ')
for f = 1:Nf
  disp(['Time-average ',e_name{f},' RMSE / spread: ',num2str(mea(f)), ' / ',num2str(sigd(f))])
end
disp(' ')

%return

% Save data
ofile = ['DATA/err_stats_obbias_',num2str(obbias),'_modbias_',num2str(modbias),'.mat'];
fid = fopen(ofile,'w');
disp(['Saving data to ',ofile])
save(ofile,'prior_mean','post_mean','prior_spread','post_spread','innov_mean','anal_incr');
fclose(fid);

% Save plot
set(gca,'ylim',[0,5])
set(fig,'PaperSize',[12,5])
set(fig,'PaperPositionMode','manual'); 
set(fig,'PaperPosition',[0,0.1,12,5]);
ofile = ['FIGS/rmse_sprd_',num2str(sig_y),'_modbias_',num2str(modbias),'.pdf'];
disp(['Saving figure to ',ofile])
saveas(fig, ofile,'pdf')

delete(gcp('nocreate'))
