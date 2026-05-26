function [th_est,phi_est,err] = find_gridded_doa(rec, amed, grid_width)

    
if nargin<3
    grid_width = 1; % in degrees
end

nrmic = size(rec,1);
nrmeas = size(amed,1);

  
   
gr_rad = grid_width/360*2*pi;

theta = (-pi:gr_rad:pi)';
phi = 0:gr_rad:pi/2;

M = length(theta);
N = length(phi);
temp = zeros(M,N,3);

temp(:,:,1) = cos(phi).*cos(theta);
temp(:,:,2) = cos(phi).*sin(theta);
temp(:,:,3) = sin(phi).*ones(M,1);

projs = squeeze(sum(reshape(temp,[M,N,3,1]).*reshape(rec',[1,1,3,nrmic]),3));
nrpairs = nchoosek(nrmic,2);

temp = zeros(nrpairs,M,N);
count = 0;
for iii = 1:nrmic-1
        for jjj = iii+1:nrmic
        count = count + 1;
        temp(count,:,:) = projs(:,:,jjj) - projs(:,:,iii);
        %temp = temp - reshape(squeeze(amed(kkk,:)),[nrpairs 1 1]);
        end
    end


th_est = zeros(nrmeas,1);
phi_est = zeros(nrmeas,1);
err = zeros(nrmeas,1);

for kkk = 1:nrmeas
   % disp(kkk)
    
    
    res_of_angle = squeeze(median(abs(temp-amed(kkk,:)')));

    [mini,indy] = min(res_of_angle(:));
    [ii,jj] = ind2sub([M,N],indy);
    
    th_est(kkk) = theta(ii);
    phi_est(kkk) = phi(jj);
    err(kkk) = mini;

end
