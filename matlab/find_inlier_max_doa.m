function [vv_est, vv_ls_est,maxinliers,err_ls] = find_inlier_max_doa(recdiff, amed, ep, do_ls, pos_z)

if nargin<3
    ep = 0.5;
end
if nargin<4
    do_ls = 1;
end

if nargin<5
    pos_z = 0;
end


nrmeas = size(amed,1);
vv_est = zeros(nrmeas,3);
maxinliers = zeros(nrmeas,1);
if do_ls
    vv_ls_est = zeros(nrmeas,3);
    err_ls = zeros(nrmeas,1);
else
    vv_ls_est = -1;
end



sz = size(recdiff,1);
sz2 = nchoosek(sz,2);

r1 = zeros(sz2,3);
r2 = zeros(sz2,3);

count = 0;

for iii = 1:sz-1
    for jjj = iii+1:sz
        count = count+1;
        r1(count,:) = recdiff(jjj,:);
        r2(count,:) = recdiff(iii,:);
    end
end

for kkk = 1:nrmeas
    t1 = zeros(sz2,1);
    t2 = zeros(sz2,1);

    count = 0;

    for iii = 1:sz-1
        for jjj = iii+1:sz
            count = count+1;
            t1(count) = amed(kkk,jjj);
            t2(count) = amed(kkk,iii);
        end
    end

    sol = two_sol4(r1,t1,r2,t2,ep);
    resok = sum(abs(sol*recdiff'-amed(kkk,:))<=ep,2);
    if pos_z
        resok(sol(:,3)<0)=0;
    end

    [nrins,besti] = max(resok);
    vv = sol(besti,:);
    vv_est(kkk,:) = vv;
    maxinliers(kkk) = nrins;
    inliers = abs(vv*recdiff'-amed(kkk,:))<=ep;
    rd_in = recdiff(inliers,:);
    tau_in = amed(kkk,inliers);

    if do_ls
        [vv_ls,mini] = least_square(rd_in,tau_in);
        err_ls(kkk) = mini;
        if mini>0
            vv_ls_est(kkk,:) = vv_ls;
        else
            vv_ls_est(kkk,:) = vv;
        end
    end
end

