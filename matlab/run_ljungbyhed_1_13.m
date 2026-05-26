%%
bp = '/my/path/to/data/ljungbyhed_audio_dataset/';

folders = {'01_flower/','02_altitudes/','03_altitudes_runway/','04_diagonals/','05_runway_length/','06_manual_close/',...
    '07_manual_far/','08_flower2/','09_range_test/','10_flower3/','11_fixed_flower/','12_big_flower/','13_big_altitudes/'};
nrfolders = length(folders);
grid_width = 0.5;
ep = 0.1;
do_ls = 1;
pos_z = 1;

okins = 30; % minimum number of inliers
degbnd = 20/360*2*pi;  % error inlier bound


%%
errs = zeros(nrfolders,12); % ls_l2_th max_l2_th disc_l2_th  ls_l1_th max_l1_th disc_l1_th   ls_l2_phi max_l2_phi disc_l2_phi  ls_l1_phi max_l1_phi disc_l1_phi
errs_ls = cell(nrfolders,1);
max_inliers = cell(nrfolders,1);
errs_disc = cell(nrfolders,1);
alltimes = zeros(nrfolders,3);
success_rate = zeros(nrfolders,4);


for iii = 1:nrfolders
    load([bp folders{iii} 'input_matlab.mat']);
    disp(folders{iii})

    disp('Running discretization doa')
    tic
    [th_est,phi_est,err_disc] = find_gridded_doa(rec, amed, grid_width);
    alltimes(iii,1) = toc;


    disp('Running inliers maximization doa')
    tic
    [vv_est, vv_ls_est,inliers,err_ls] = find_inlier_max_doa(recdiff, amed, ep, do_ls,pos_z);
    alltimes(iii,2) = toc;


    disp('Running LS doa')
    tic
    vv_ls0 = find_ls_doa(recdiff, amed);
    alltimes(iii,3) = toc;
    errs_ls{iii} = err_ls;
    max_inliers{iii} = inliers;
    errs_disc{iii} = err_disc;
    [th_max,phi_max] = v_to_angle(vv_est);
    [th_ls0,phi_ls0] = v_to_angle(vv_ls0);
    [th_ls,phi_ls] = v_to_angle(vv_ls_est);
    [~, startid] = min(abs(angle_est_time-gt_time(1)));
    [~, endid] = min(abs(angle_est_time-gt_time(end)));
    id = startid:endid;
    sf = 360/2/pi;
    ok =  max_inliers{iii}(id) > okins;

    figure(1)
    clf
    ll = plot(gt_time,theta_gt*sf,'-');
    set(ll,'LineWidth',2)
    hold on
    ll = plot(angle_est_time(id),th_est(id)*sf,'x');
    set(ll,'MarkerSize',5);
    set(ll,'LineWidth',2)
    ll = plot(angle_est_time(id),th_max(id)*sf,'.');
    set(ll,'MarkerSize',5);
    legend({'Ground truth','Discretized','Inlier maximization'})
    xlabel('Time (s)')
    ylabel('Azimuth angle (deg)')
    axis([50 425 -185 185])

    figure(2)
    clf
    ll = plot(gt_time,phi_gt*sf,'-');
    set(ll,'LineWidth',2)
    hold on
    ll = plot(angle_est_time(id),phi_est(id)*sf,'x');
    set(ll,'MarkerSize',5);
    set(ll,'LineWidth',2)
    ll = plot(angle_est_time(id),phi_max(id)*sf,'.');
    set(ll,'MarkerSize',5);
    legend({'Ground truth','Discretized','Inlier maximization'})
    xlabel('Time (s)')
    ylabel('Elevation angle (deg)')
    axis([50 425 -5 100])
    drawnow
 
    [gt_time,uni_time] = unique(gt_time);
    theta_gt = theta_gt(uni_time);
    phi_gt = phi_gt(uni_time);



    theta_gt_interp = interp1(gt_time,theta_gt,angle_est_time(id));
    phi_gt_interp = interp1(gt_time,phi_gt,angle_est_time(id));

    res_th_ls = theta_gt_interp-th_ls(id)';
    res_th_max = theta_gt_interp-th_max(id)';
    res_th_disc = theta_gt_interp-th_est(id)';
    res_th_ls0 = theta_gt_interp-th_ls0(id)';

    res_phi_ls = phi_gt_interp-phi_ls(id)';
    res_phi_max = phi_gt_interp-phi_max(id)';
    res_phi_disc = phi_gt_interp-phi_est(id)';
    res_phi_ls0 = phi_gt_interp-phi_ls0(id)';



    res_th_ls = res_th_ls(2:end-1);
    res_th_max = res_th_max(2:end-1);
    res_th_disc = res_th_disc(2:end-1);
    res_th_ls0 = res_th_ls0(2:end-1);


    res_phi_ls = res_phi_ls(2:end-1);
    res_phi_max = res_phi_max(2:end-1);
    res_phi_disc = res_phi_disc(2:end-1);
    res_phi_ls0 = res_phi_ls0(2:end-1);

    success_rate(iii,1) = sum(abs(res_phi_ls)<degbnd & abs(res_th_ls)<degbnd)/length(res_th_ls);
    success_rate(iii,2) = sum(abs(res_phi_max)<degbnd & abs(res_th_max)<degbnd)/length(res_th_ls);
    success_rate(iii,3) = sum(abs(res_phi_disc)<degbnd & abs(res_th_disc)<degbnd)/length(res_th_ls);
    success_rate(iii,4) = sum(abs(res_phi_ls0)<degbnd & abs(res_th_ls0)<degbnd)/length(res_th_ls);


    errs(iii,:) = [norm(res_th_ls) norm(res_th_max) norm(res_th_disc) sum(abs(res_th_ls)) sum(abs(res_th_max)) sum(abs(res_th_disc)) ...
        norm(res_phi_ls) norm(res_phi_max) norm(res_phi_disc) sum(abs(res_phi_ls)) sum(abs(res_phi_max)) sum(abs(res_phi_disc))];
    disp(errs(iii,:));

end



%% check start and end time for gt

for iii = 1:nrfolders

    load([bp folders{iii} 'input_matlab.mat']);
    disp(folders{iii})
    figure(1)
    clf
    ll = plot(gt_time,theta_gt,'-');
    set(ll,'LineWidth',2)
    figure(2)
    clf
    ll = plot(gt_time,phi_gt,'-');
    set(ll,'LineWidth',2)
    drawnow
    pause

end
