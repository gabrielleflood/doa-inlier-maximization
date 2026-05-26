%%
bp = '/Users/magnuso/Jobbmapp/projekt/audiovisual_drone_detection-main/data/ljungbyhed_audio_dataset/';
folders = {'01_flower/','02_altitudes/','03_altitudes_runway/','04_diagonals/','05_runway_length/','06_manual_close/',...
    '07_manual_far/','08_flower2/','09_range_test/','10_flower3/','11_fixed_flower/','12_big_flower/','13_big_altitudes/'};
nrfolders = length(folders);
grid_width = 0.5;
ep = 0.1;
do_ls = 1;
pos_z = 1;

okins = 30;
iii = 1;
load([bp folders{iii} 'input_matlab.mat']);
disp(folders{iii})

%%
%grid_widths = [0.1 0.2 0.5 1 2 5 10];
%grid_widths = [0.1 0.3 0.5 0.7 0.9  1.1  1.3  1.5  1.7  1.9  2 2.3 2.7 3 3.5 4 5 6 7 8 9 10];

grid_widths = linspace(0.1,10,40);


nn = length(grid_widths);
grid_times = zeros(1,nn);
nrlocs = 100;

loctimes = zeros(1,nrlocs);
disp('Running discretization doa')
for iii = 1:nn
    grid_width = grid_widths(iii);
    disp(grid_width)

   
for kkk = 1:nrlocs   
tic
%[th_est,phi_est,err_disc] = find_gridded_doa(rec, amed((2001:10:3000)+kkk,:), grid_width);
[th_est,phi_est,err_disc] = find_gridded_doa(rec, amed(randperm(4000,100),:), grid_width);
tti = toc/100;
loctimes(kkk) = tti;
end
%grid_times(iii) = tti;
%grid_times(iii) = exp(median(log(loctimes)));
grid_times(iii) = median(loctimes);
%grid_times(iii) = min(loctimes);


%disp(loctimes)
end


%%
disp('Running inliers maximization doa')
nrmics = [12 10 8 6 5 4 ];
nn = length(nrmics);
inliermax_times = zeros(1,nn);

ids = [];
for iii = 1:11
    for jjj = iii+1:12
        ids = [ids;iii jjj];
    end
end

for iii = 1:nn
    nrmic = nrmics(iii);
    disp(nrmic)
    subids = find(ids(:,1)<=nrmic & ids(:,2)<=nrmic);

tic
[vv_est, vv_ls_est,inliers,err_ls] = find_inlier_max_doa(recdiff(subids,:), amed(1001:2000,subids), ep, do_ls,pos_z);
tti = toc/1000;
inliermax_times(iii) = tti;

end


%%
%x1 = grid_widths(1:7);
x1 = grid_widths;

x2 = nrmics;
%y1 = grid_times(1:7);

y1 = grid_times;
%y1 = exp(medfilt1(log(grid_times),7));
%y1 = exp(conv(log(grid_times),ones(1,3)/3,'same'));


y2 = inliermax_times;

%%

figure(1)
clf


%hSc=plot(x1,log10(y1),'-');
hSc=semilogy(x2,y2,'-');
set(hSc,'LineWidth',2)
hold on
hAx=gca;            % save first axes handle..
hAx(2)=axes('position',hAx(1).Position, ... % create the second axis, top/right...
            'color','none', ...
            'XAxisLocation','top', ...
            'YAxisLocation','right');
%ylim(hAx,[-5 0])
%ylim(hAx,[1e-8 0.5])


hold on
%hSc(2)=plot(hAx(2),x2,log10(y2),'-');
hSc(2)=semilogy(hAx(2),x1,y1,'-');
set(hSc(2),'LineWidth',2)

%ylim(hAx,[1e-5 1e0])



%%
figure(1)
clf
ll = semilogy(x1,y1);
set(ll,'LineWidth',2)
hold on
ll = semilogy(x2-3,y2);
set(ll,'LineWidth',2)
ll = legend({'Using discretization','Using inlier maximization'});

set(ll,'FontSize',15)
xlabel('Grid width (deg)')
ylabel('Execution time (s)')
grid on
hAx=gca;            % save first axes handle..
hAx(2)=axes('position',hAx(1).Position, ... % create the second axis, top/right...
            'color','none', ...
            'XAxisLocation','top', ...
            'YAxisLocation','right');
ylim(hAx,[0.5e-4 1])  
hold on
% hSc(2)=semilogy(3+0*x1,y1,'.');
% hSc(2)=semilogy(13+0*x2,y2,'.');
hSc(2)=plot([3 13],[1 1],'.');
xlabel('Number of microphones')
% ylim(hAx(1),[0.5e-4 1])  
% ylim(hAx(2),[0 1])  
xlim(hAx(2),[3 13]) 
set(hAx(2),'ytick',[])




%% test 
figure(1)
%clf
ll = semilogy(x1,y1);
set(ll,'LineWidth',2)
hold on
ll = semilogy(x2-3,y2);
set(ll,'LineWidth',2)
ll = legend({'Using discretization','Using inlier maximization'});

set(ll,'FontSize',15)
xlabel('Grid width (deg)')
ylabel('Execution time (s)')
grid on
