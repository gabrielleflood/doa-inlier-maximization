function [th,phi] = v_to_angle(vv)

a = vv(:,1);
b = vv(:,2);
c = vv(:,3);

th = atan2(b,a);
phi = atan2(c,sqrt(a.^2+b.^2));

