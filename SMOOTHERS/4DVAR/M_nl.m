% This function integrates the Lorenz-96 equations forward in time 
% using the fourth order Runge-Kutta scheme
%
% Input:
%        x <- vector of size n
%       dt <- time step
%        T <- integration time
%        F <- forcing term

function x = M_nl(x,dt,T,F)

  Nx = length(x(:,1));

  % Create buffer zones on x for periodic domain
  x = [x(Nx-1,1); x(Nx,1); x(:,1); x(1,1)];

  for t = 1:T

    % Place variables in vectors
    y1 = x(4:end); 
    y2 = x(3:end-1);
    y3 = x(2:end-2);
    y4 = x(1:end-3);

    % Update variables
    x(3:end-1) = y2 + ( (y1-y4).*y3 - y2 + F )*dt*(6-3*dt+dt*dt-dt*dt*dt/4)/6;

    % Update buffer zones
    x(1:2) = x(end-2:end-1);
    x(end) = x(3);

  end

  % Remove buffer zones
  x(1:2) = [];
  x(end) = [];

end
