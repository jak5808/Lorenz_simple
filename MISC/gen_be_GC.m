% This function generates an m x m covariance matrix based on a Guassian correlation model

% Input:   
%       sigma <- standard deviation for the Gaussian correlation model
%         amp <- variance at each grid point (assumed constant at all points)
%           m <- length of the domain

function B = gen_be_periodic(b,n);

% n <- domain size
% x <- peak of correlation function
% b <- length scale parameter

for x = 1:n

  clear coef

  for i=1:n
    z=min(abs(i-x),n-abs(i-x));
    r=z/b;
    if(z >= 2*b)
        coef(i)=0.0;
    elseif (z >= b && z < 2*b)
        coef(i)=((((r/12.-0.5)*r+0.625)*r+5./3.)*r-5.)*r+4.-2./(3.*r);
    else
        coef(i)=(((-0.25*r+0.5)*r+0.625)*r-5./3.)*r^2+1.;
    end
  end

  B(x,:) = coef;

end
