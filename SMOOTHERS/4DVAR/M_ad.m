% This function performs the adjoint model integration for
% the Lorenz-96 model using the fourth order Runge-Kutta scheme
%
% Input:
%        x <- vector of size n
%       xb <- vector of size T x n
%       dt <- time step
%        T <- integration time
%        F <- forcing term

function x = M_ad(x,xb,dt,T)

  Nx = length(x(:,1));

  % Create buffer zones on x for periodic domain
  x  = [ x(Nx,1); x(:,1); x(1,1); x(2,1) ];
  xb = [ xb(Nx-1,:); xb(Nx,:); xb(:,:); xb(1,:); xb(2,:) ];

  % Fill x with zeros from t = 1:T
  x = [zeros(Nx+3,T),x];

  for t = T+1:-1:2

    % Place variables and background states in vectors
    y1 = x(4:end  ,t); x1 = xb(5:end  ,t);
    y2 = x(3:end-1,t); x2 = xb(4:end-1,t);
    y3 = x(2:end-2,t); x3 = xb(2:end-3,t);
    y4 = x(1:end-3,t); x4 = xb(1:end-4,t);

    % Update variables
    x(2:end-2,t-1) = y3 + ( x4.*y4 - y3 + (x1-x3).*y2 - x2.*y1 )*dt*(6-3*dt+dt*dt-dt*dt*dt/4)/6;

    % Update buffer zones
    x(1,t-1) = x(end-2,t-1);
    x(end-1:end,t-1) = x(2:3,t-1);

  end

  % Remove buffer zones
  x(1,:) = [];
  x(end-1:end,:) = [];

end
