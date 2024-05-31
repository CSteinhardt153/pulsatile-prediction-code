%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%slope_analysis.m
%Compares slope results on experimental and real data
%Last updated 10.13.23 CRS
%========================================================
%% SLOPE ANALYSES FOR COMPARISON WITH EXPERIMENTAL DATA:
%ID and compare performance
cd('..');
base_dir = pwd;
data_dir = fullfile(base_dir,'relevant_data');

perS_dat = dir(fullfile(data_dir ,'pr_fr_S*_01-15-2023.mat'))
load(fullfile(data_dir,perS_dat(1).name),'per_S')
cur_dir = data_dir%;'/Users/thia/Dropbox/Postdoc/pulse_stim_follow_up';
clear slopes_per_S
%check slopes ever pr_dff steps in simulated ata
pr_diff= 10;%%25;%%14;%12;
pr_range= 1:pr_diff:150;
I_range = 5:5:40;

for n_S = 1:7
    q= load(fullfile(cur_dir,perS_dat(n_S).name));
    I_vals = q.per_S.I(I_range )*-20;
    tmp = strsplit(perS_dat(n_S).name,'_01');
    tmp2= strsplit(tmp{1},'_S');

    pulse_diff= unique(diff(q.per_S.pulse_rate(pr_range)));
    pr_dfr = q.per_S.pr_fr_dat - q.per_S.pr_fr_dat(:,:,1);
    figure(1);subplot(1,7,n_S);
    histogram((diff(squeeze(mean(pr_dfr(:,pr_range,I_range )))))/pulse_diff,[-2:.1:1]);
    title(['S=' num2str(tmp2{2})])
    slopes_per_S(n_S,:,:) = (diff(squeeze(mean(q.per_S.pr_fr_dat(:,pr_range,I_range))))/pulse_diff);
    xlabel('Slope (sps/pps)'); xlim([-2 1])
    set(gca,'fontsize',15)
    S_ord(n_S) = mean(q.per_S.pr_fr_dat(:,1,1));
end
figure(2); histogram(slopes_per_S(:));
xlabel('Slope (sps/pps)');
from_code= 'I_S_fit_2_21_23.m';
set(gca,'fontsize',15); title('Overall Slopes across Population');
%save('slopes_per_S_7_03_23','slopes_per_S','perS_dat','I_range','pr_range','from_code','S_ord')

%% B&W plot of pr v. fr/ slope
%Supplemental Figure 3A
use_idxs = 1:2:length(pr_range)-1;
figure(3);
I_cols=repmat(linspace(.65,0,length(I_vals)-1)',[1 3])% Current color map
cent_pr= unique(diff(q.per_S.pulse_rate(pr_range(use_idxs )))/2)+ q.per_S.pulse_rate(pr_range(use_idxs ));%(q.per_S.pulse_rate(1:pr_diff:150));

%I_range = 5:6:44;
plt_ord = [1 7 3 4 5 2 6];

sim_data = []
for n_S = 1:7

    q= load(fullfile(cur_dir,perS_dat(n_S).name));

    subplot(2,7,plt_ord(n_S));
    tmp = strsplit(perS_dat(n_S).name,'_01');
    tmp2= strsplit(tmp{1},'_S');

    for n_I = 1:(length(I_vals)-1)
        plot(q.per_S.pulse_rate,...
            mean(q.per_S.pr_fr_dat(:,:,I_range(n_I)),1)-mean(q.per_S.pr_fr_dat(:,:,I_range(1)),1),...
            'color',[I_cols(n_I,:) ],'markersize',13); hold on;

        errorbar(q.per_S.pulse_rate(pr_range(use_idxs )),...
            mean(q.per_S.pr_fr_dat(:,pr_range(use_idxs ),I_range(n_I)),1)-...
            mean(q.per_S.pr_fr_dat(:,pr_range(use_idxs ),I_range(1)),1),...
            std(q.per_S.pr_fr_dat(:,pr_range(use_idxs ),I_range(n_I)),[],1)/sqrt(10),...
            '.','color',I_cols(n_I,:),'markersize',13); hold on;
    
        tmp_sim = [repmat([str2num(tmp2{2}), I_range(n_I)],size(q.per_S.pulse_rate(pr_range(use_idxs ))')),...
            q.per_S.pulse_rate(pr_range(use_idxs ))',...
            (mean(q.per_S.pr_fr_dat(:,pr_range(use_idxs ),I_range(n_I)),1)-...
            mean(q.per_S.pr_fr_dat(:,pr_range(use_idxs ),I_range(1)),1))',...
            std(q.per_S.pr_fr_dat(:,pr_range(use_idxs ),I_range(n_I)),[],1)'/sqrt(10)]
        sim_data = vertcat(sim_data,tmp_sim);
    end
    set(gca,'fontsize',15);
    
    ylabel('Firing Rate (sps)'); xlim([0 300])
    box off; title(['S = ' num2str(tmp2{2})]);
    % if n_S >= 5
    %     ylim([-50 100])
    % end
    subplot(2,7,plt_ord(n_S)+7);
    for n_I = 1:(length(I_vals)-1)
        plot(cent_pr(1:end),squeeze(slopes_per_S(n_S,(use_idxs ),n_I)),'.-', ...
            'color',I_cols(n_I,:),'markersize',13); hold on;

    end
    if n_S >= 5
        ylim([-0.75 1]);
    else
        ylim([-2 1]); box off;
    end
    set(gca,'fontsize',15);
    ylabel('Slope (sps/pps)')
    xlabel('Pulse Rate (pps)'); box off;

end

