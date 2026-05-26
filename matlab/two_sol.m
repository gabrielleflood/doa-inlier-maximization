function [sol1,sol2] = two_sol(r1,t1,r2,t2,ep1,ep2)
    r1x = r1(:,1);
    r1y = r1(:,2);
    r1z = r1(:,3);
    r2x = r2(:,1);
    r2y = r2(:,2);
    r2z = r2(:,3);
    r1x2 = r1x.*r1x;
    r1y2 = r1y.*r1y;
    r1z2 = r1z.*r1z;
    r2x2 = r2x.*r2x;
    r2y2 = r2y.*r2y;
    r2z2 = r2z.*r2z;
    ep12 = ep1.*ep1;
    ep22 = ep2.*ep2;
    t12 = t1.*t1;
    t22 = t2.*t2;

    f1 = abs(- ep12.*r2x2 - ep12.*r2y2 - ep12.*r2z2 + 2.*ep1.*ep2.*r1x.*r2x + 2.*ep1.*ep2.*r1y.*r2y + 2.*ep1.*ep2.*r1z.*r2z + 2.*ep1.*r1x.*r2x.*t2 - 2.*ep1.*r2x2.*t1 + 2.*ep1.*r1y.*r2y.*t2 - 2.*ep1.*r2y2.*t1 + 2.*ep1.*r1z.*r2z.*t2 - 2.*ep1.*r2z2.*t1 - ep22.*r1x2 - ep22.*r1y2 - ep22.*r1z2 - 2.*ep2.*r1x2.*t2 + 2.*ep2.*r1x.*r2x.*t1 - 2.*ep2.*r1y2.*t2 + 2.*ep2.*r1y.*r2y.*t1 - 2.*ep2.*r1z2.*t2 + 2.*ep2.*r1z.*r2z.*t1 + r1x2.*r2y2 + r1x2.*r2z2 - r1x2.*t22 - 2.*r1x.*r2x.*r1y.*r2y - 2.*r1x.*r2x.*r1z.*r2z + 2.*r1x.*r2x.*t1.*t2 + r2x2.*r1y2 + r2x2.*r1z2 - r2x2.*t12 + r1y2.*r2z2 - r1y2.*t22 - 2.*r1y.*r2y.*r1z.*r2z + 2.*r1y.*r2y.*t1.*t2 + r2y2.*r1z2 - r2y2.*t12 - r1z2.*t22 + 2.*r1z.*r2z.*t1.*t2 - r2z2.*t12);
      
    q11 = (ep1.*r2y - ep2.*r1y - r1y.*t2 + r2y.*t1)./(r1x.*r2y - r2x.*r1y) + ((r1y.*r2z - r2y.*r1z).*(ep1.*r2x2.*r1z + ep2.*r1x2.*r2z + ep1.*r2y2.*r1z + ep2.*r1y2.*r2z + r2x2.*r1z.*t1 + r1x2.*r2z.*t2 + r2y2.*r1z.*t1 + r1y2.*r2z.*t2 + r1x.*r2y.*sqrt(f1) - r2x.*r1y.*sqrt(f1) - ep1.*r1x.*r2x.*r2z - ep2.*r1x.*r2x.*r1z - ep1.*r1y.*r2y.*r2z - ep2.*r1y.*r2y.*r1z - r1x.*r2x.*r1z.*t2 - r1x.*r2x.*r2z.*t1 - r1y.*r2y.*r1z.*t2 - r1y.*r2y.*r2z.*t1))./((r1x.*r2y - r2x.*r1y).*(r1x2.*r2y2 + r1x2.*r2z2 - 2.*r1x.*r2x.*r1y.*r2y - 2.*r1x.*r2x.*r1z.*r2z + r2x2.*r1y2 + r2x2.*r1z2 + r1y2.*r2z2 - 2.*r1y.*r2y.*r1z.*r2z + r2y2.*r1z2));
    q12 = (ep1.*r2y - ep2.*r1y - r1y.*t2 + r2y.*t1)./(r1x.*r2y - r2x.*r1y) + ((r1y.*r2z - r2y.*r1z).*(ep1.*r2x2.*r1z + ep2.*r1x2.*r2z + ep1.*r2y2.*r1z + ep2.*r1y2.*r2z + r2x2.*r1z.*t1 + r1x2.*r2z.*t2 + r2y2.*r1z.*t1 + r1y2.*r2z.*t2 - r1x.*r2y.*sqrt(f1) + r2x.*r1y.*sqrt(f1) - ep1.*r1x.*r2x.*r2z - ep2.*r1x.*r2x.*r1z - ep1.*r1y.*r2y.*r2z - ep2.*r1y.*r2y.*r1z - r1x.*r2x.*r1z.*t2 - r1x.*r2x.*r2z.*t1 - r1y.*r2y.*r1z.*t2 - r1y.*r2y.*r2z.*t1))./((r1x.*r2y - r2x.*r1y).*(r1x2.*r2y2 + r1x2.*r2z2 - 2.*r1x.*r2x.*r1y.*r2y - 2.*r1x.*r2x.*r1z.*r2z + r2x2.*r1y2 + r2x2.*r1z2 + r1y2.*r2z2 - 2.*r1y.*r2y.*r1z.*r2z + r2y2.*r1z2));
    q21 = - (ep1.*r2x - ep2.*r1x - r1x.*t2 + r2x.*t1)./(r1x.*r2y - r2x.*r1y) - ((r1x.*r2z - r2x.*r1z).*(ep1.*r2x2.*r1z + ep2.*r1x2.*r2z + ep1.*r2y2.*r1z + ep2.*r1y2.*r2z + r2x2.*r1z.*t1 + r1x2.*r2z.*t2 + r2y2.*r1z.*t1 + r1y2.*r2z.*t2 + r1x.*r2y.*sqrt(f1) - r2x.*r1y.*sqrt(f1) - ep1.*r1x.*r2x.*r2z - ep2.*r1x.*r2x.*r1z - ep1.*r1y.*r2y.*r2z - ep2.*r1y.*r2y.*r1z - r1x.*r2x.*r1z.*t2 - r1x.*r2x.*r2z.*t1 - r1y.*r2y.*r1z.*t2 - r1y.*r2y.*r2z.*t1))./((r1x.*r2y - r2x.*r1y).*(r1x2.*r2y2 + r1x2.*r2z2 - 2.*r1x.*r2x.*r1y.*r2y - 2.*r1x.*r2x.*r1z.*r2z + r2x2.*r1y2 + r2x2.*r1z2 + r1y2.*r2z2 - 2.*r1y.*r2y.*r1z.*r2z + r2y2.*r1z2));
    q22 = - (ep1.*r2x - ep2.*r1x - r1x.*t2 + r2x.*t1)./(r1x.*r2y - r2x.*r1y) - ((r1x.*r2z - r2x.*r1z).*(ep1.*r2x2.*r1z + ep2.*r1x2.*r2z + ep1.*r2y2.*r1z + ep2.*r1y2.*r2z + r2x2.*r1z.*t1 + r1x2.*r2z.*t2 + r2y2.*r1z.*t1 + r1y2.*r2z.*t2 - r1x.*r2y.*sqrt(f1) + r2x.*r1y.*sqrt(f1) - ep1.*r1x.*r2x.*r2z - ep2.*r1x.*r2x.*r1z - ep1.*r1y.*r2y.*r2z - ep2.*r1y.*r2y.*r1z - r1x.*r2x.*r1z.*t2 - r1x.*r2x.*r2z.*t1 - r1y.*r2y.*r1z.*t2 - r1y.*r2y.*r2z.*t1))./((r1x.*r2y - r2x.*r1y).*(r1x2.*r2y2 + r1x2.*r2z2 - 2.*r1x.*r2x.*r1y.*r2y - 2.*r1x.*r2x.*r1z.*r2z + r2x2.*r1y2 + r2x2.*r1z2 + r1y2.*r2z2 - 2.*r1y.*r2y.*r1z.*r2z + r2y2.*r1z2));
    q31 = (ep1.*r2x2.*r1z + ep2.*r1x2.*r2z + ep1.*r2y2.*r1z + ep2.*r1y2.*r2z + r2x2.*r1z.*t1 + r1x2.*r2z.*t2 + r2y2.*r1z.*t1 + r1y2.*r2z.*t2 + r1x.*r2y.*sqrt(f1) - r2x.*r1y.*sqrt(f1) - ep1.*r1x.*r2x.*r2z - ep2.*r1x.*r2x.*r1z - ep1.*r1y.*r2y.*r2z - ep2.*r1y.*r2y.*r1z - r1x.*r2x.*r1z.*t2 - r1x.*r2x.*r2z.*t1 - r1y.*r2y.*r1z.*t2 - r1y.*r2y.*r2z.*t1)./(r1x2.*r2y2 + r1x2.*r2z2 - 2.*r1x.*r2x.*r1y.*r2y - 2.*r1x.*r2x.*r1z.*r2z + r2x2.*r1y2 + r2x2.*r1z2 + r1y2.*r2z2 - 2.*r1y.*r2y.*r1z.*r2z + r2y2.*r1z2);
    q32 = (ep1.*r2x2.*r1z + ep2.*r1x2.*r2z + ep1.*r2y2.*r1z + ep2.*r1y2.*r2z + r2x2.*r1z.*t1 + r1x2.*r2z.*t2 + r2y2.*r1z.*t1 + r1y2.*r2z.*t2 - r1x.*r2y.*sqrt(f1) + r2x.*r1y.*sqrt(f1) - ep1.*r1x.*r2x.*r2z - ep2.*r1x.*r2x.*r1z - ep1.*r1y.*r2y.*r2z - ep2.*r1y.*r2y.*r1z - r1x.*r2x.*r1z.*t2 - r1x.*r2x.*r2z.*t1 - r1y.*r2y.*r1z.*t2 - r1y.*r2y.*r2z.*t1)./(r1x2.*r2y2 + r1x2.*r2z2 - 2.*r1x.*r2x.*r1y.*r2y - 2.*r1x.*r2x.*r1z.*r2z + r2x2.*r1y2 + r2x2.*r1z2 + r1y2.*r2z2 - 2.*r1y.*r2y.*r1z.*r2z + r2y2.*r1z2);
    sol1 = [q11,q21,q31];
    sol2 = [q12,q22,q32];
 

    

    