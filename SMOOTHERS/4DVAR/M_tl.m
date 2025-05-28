% This function performs the tangent linear model integration for
% the Lorenz-96 model using the fourth order Runge-Kutta scheme
%
% Input:
%        x <- vector of size n
%       xb <- vector of size T x n
%       dt <- time step
%        T <- integration time

function x = M_tl(x,xb,dt,T)

  Nx = length(x(:,1));

  % Create buffer zones on x for periodic domain
  x  = [x(Nx-1,1); x(Nx,1); x(:,1); x(1,1) ];
  xb = [xb(Nx-1,:); xb(Nx,:); xb(:,:); xb(1,:)];

  for t = 1:T

    % Place variables in vectors

    y1 = x(4:end  ,t); x1 = xb(4:end  ,t);
    y2 = x(3:end-1,t); x2 = xb(2:end-2,t);
    y3 = x(2:end-2,t); x3 = xb(1:end-3,t);
    y4 = x(1:end-3,t); 

    % Update variables
    x(3:end-1,t+1) = y2 + ( (x1-x3).*y3 + x2.*y1 - y2 - x2.*y4 )*dt*(6-3*dt+dt*dt-dt*dt*dt/4)/6;

    % Update buffer zones
    x(1:2,t+1) = x(end-2:end-1,t+1);
    x(end,t+1) = x(3,t+1);

  end

  % Remove buffer zones
  x(1:2,:) = [];
  x(end,:) = [];

end
