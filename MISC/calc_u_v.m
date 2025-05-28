function [u,v] = calc_u_v(vt,vr,IO,JO)

KM = 1;
[IM,JM,KM] = size(vt);
if ~exist('IO') & ~exist('JO'),
  IO = floor(IM/2) + 1;
  JO = floor(JM/2) + 1;
end

for k = 1:KM
  for j = 1:JM
    for i = 1:IM

       x = i - IO;
       y = j - JO;
       
       phi = atan(y/x);
       if(x < 0), phi = phi + pi; end

       u(i,j,k) = vr(i,j,k)*cos(phi) - vt(i,j,k)*sin(phi);
       v(i,j,k) = vt(i,j,k)*cos(phi) + vr(i,j,k)*sin(phi);

    end
  end
end

u(isnan(u)) = 0;
v(isnan(v)) = 0;
