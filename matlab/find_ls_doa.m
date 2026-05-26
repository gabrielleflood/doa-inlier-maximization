function [vv_ls0, err_ls] = find_ls_doa(recdiff, amed)


nrmeas = size(amed,1);
vv_ls0 = zeros(nrmeas,3);
err_ls = zeros(nrmeas,1);

for kkk = 1:nrmeas
    [vv_ls,mini] = least_square(recdiff,amed(kkk,:));
    vv_ls0(kkk,:) = vv_ls;
    err_ls(kkk) = mini;
end

