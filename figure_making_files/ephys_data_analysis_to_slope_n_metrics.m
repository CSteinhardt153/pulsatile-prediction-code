%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ephys_data_analysis_to_slope_n_metrics.m
%Continuation of analysis in data_analysis_comp_model.m
%%% Analyzing the 3 sets of data from mitchell_data folder for
%%%%%% evidence of bends in pulse amp/rate fr relationship
%%%%%% amplitude effects across neurons
%%%%%% "state" of neuron influencing level of bends
%%%%%% Slope calculations and stastic metrics versus the simulated data.

%Last Updated 10.14.23 CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd('..');
base_dir = pwd;
data_dir = fullfile(base_dir,'relevant_data');
expt_data_dir = fullfile(data_dir,'mitchell_data_all');
addpath(fullfile(genpath(base_dir),'helper_functions'))
%cd(data_dir);

%%
%addpath('/Users/cynthiasteinhardt/Dropbox/single-neuron-stim-model/vestibular-neuron-models/vest_model_pulsatile/simpler_format');
cur_range= [0 300];
% Data from different current amplitudes (2 & 3)
% Folders:
orig_data_fld = 'afferent_data';
amp_data_fld = {'afferents_data_2','afferents_data_3'};

% Data from afferent_data_updated.xls
%Filename	afferent ID	mean_FR	mean_ISI	std_ISI	CV	CV*	Classification
dat_types = {'mean_fr','mean_isi','std_isi','cv','cv*'};
afferent_info = [32.86,	30.37,	21.80,	0.72,	0.42,
    43.34	23.08	11.02	0.48	0.35
    31.37	31.98	11.01	0.34	0.18
    26.16	38.04	18.93	0.50	0.20
    21.17	46.73	9.33	0.20	0.02
    19.24	51.86	21.41	0.41	0.04
    25.34	39.45	17.88	0.45	0.16
    nan     nan     nan     nan     nan];%
%have data from aff8 but not this info from mitchell excel
is_reg = (afferent_info(:,5) < 0.1);

orig_neurs = [1:6];
cur_amp_neurs = [1 2 7 4 6 8];

%%%%% Current Amplitude Info (amp test 1)
curr_perc = [25	50	75	87.5 100];
amps_per_neur = [40	80	120	140	160
    42	84	126	nan	168
    42	84	126	nan	168];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Slope analysis on Recorded Data:
% Single amplitude data:
aff_dats = dir(fullfile(fullfile(expt_data_dir,orig_data_fld),'affer*.mat'))
aff_names = {aff_dats(:).name};
name_info  =cellfun(@(x) strsplit(x(9:end),'.'),aff_names,'UniformOutput',false);
aff_nums = unique(cellfun(@(x) str2num(x{1}),name_info));

n_plot = 21;

I_range = 0:1:400;

for n_neurs = 1:length(aff_nums)
    aff_file_names = dir(fullfile(aff_dats(1).folder,sprintf('afferent%s*',num2str(aff_nums(n_neurs)))))
    data_per_aff = load(fullfile(aff_file_names.folder,aff_file_names.name));
    min_isi(n_neurs) = min(diff(find(data_per_aff.ua)));
end
%%
for n_neurs = 1:length(aff_nums)
    aff_file_names = dir(fullfile(aff_dats(1).folder,sprintf('afferent%s*',num2str(aff_nums(n_neurs)))));
    data_per_aff = load(fullfile(aff_file_names.folder,aff_file_names.name));
    plot_it= 0;
    %%[fin_nums base_fr] = data_t_fr(data_per_aff, plot_it,n_neurs,n_neurs,length(aff_nums),afferent_info,n_plot);
    [fin_nums base_fr] = data_split(data_per_aff,plot_it,n_neurs,n_neurs,length(aff_nums),afferent_info,n_plot);

    fr_per = [base_fr(1) fin_nums.fr(:,1)'];
    pr_per = [0 fin_nums.pr(:,1)'];
    sing_resp(n_neurs).fr_slope_per_points = [diff(fr_per)./diff(pr_per)];
    sing_resp(n_neurs).pr_centers = [diff(pr_per)/2+ pr_per(1:end-1)];
    sing_resp(n_neurs).aff_num=aff_nums(n_neurs);
    sing_resp(n_neurs).fin_nums = fin_nums;
    sing_resp(n_neurs).base_fr = base_fr;
    sing_resp(n_neurs).fr_per = fr_per;
    sing_resp(n_neurs).pr_per =pr_per;

end

all_sing_slope = [sing_resp.fr_slope_per_points];
%% PLOT RELATIONSHIPS AND SLOPES
%================= Supplemental Figure 2 A/ Figure 4 D ================
n_aff = [1 3:6]; n_fig = 2; line_col = 'k';%last indicates iterating variable
plot_pfr_slopes(n_aff,sing_resp,n_fig,line_col,[2,5],[],[]) %Made general function - below for plotting these


%% DIFF AMP PER AFFERENT:
n_plot2 = 2;
tot_aff = 1;
for n_fld = 1:length(amp_data_fld)
    %cd(fullfile(data_dir,amp_data_fld{n_fld}))
    aff_dats= dir(fullfile(expt_data_dir,amp_data_fld{n_fld},'affer*.mat'));
    aff_names = {aff_dats(:).name};
    name_info  =cellfun(@(x) strsplit(x(9:end),'_'),aff_names,'UniformOutput',false);
    aff_nums = unique(cellfun(@(x) str2num(x{1}),name_info));
    
    for cur_neur= 1:length(aff_nums)
        aff_file_names = dir(fullfile(expt_data_dir,amp_data_fld{n_fld},sprintf('afferent%s*',num2str(aff_nums(cur_neur)))));
        
        clear cur_neur_amp
        for n_amps = 1:length(aff_file_names) %amps for each afferent
            tmp_curr_string = strsplit(aff_file_names(n_amps).name,'_');
            aff_num = str2num(tmp_curr_string{1}(9:end));
            
            if aff_num == 3
                aff_num = 7; % I think from the excel
            end
            
            neur_idx = find(cur_amp_neurs == aff_num);
            tmp_curr_string = strsplit(tmp_curr_string{2},'u');
            cur_neur_amp(n_amps) = str2num(tmp_curr_string{1});
            %Extract data from file
            data_per_aff = load(fullfile(expt_data_dir,amp_data_fld{n_fld},aff_file_names(n_amps).name));
            if n_fld == 2
                data_per_aff.stim = data_per_aff.stim';
            end
        %    figure(n_fld*100+cur_neur);
        %        subplot(length(aff_file_names),1,n_amps)
        %        disp(['I ' num2str(n_amps)]);
             [fin_nums base_fr] = data_split(data_per_aff,0,neur_idx,aff_num,length(cur_amp_neurs),afferent_info,n_plot2);
             
             fr_per = [base_fr(1) fin_nums.fr(:,1)'];
             pr_per = [0 fin_nums.pr(:,1)'];
             aff(tot_aff).amp_resp(n_amps).fr_slope_per_points = [diff(fr_per)./diff(pr_per)];
             aff(tot_aff).amp_resp(n_amps).pr_centers = diff(pr_per)/2+ pr_per(1:end-1);
             aff(tot_aff).aff_num = aff_num;
             aff(tot_aff).amp_resp(n_amps).I = cur_neur_amp(n_amps);
             aff(tot_aff).amp_resp(n_amps).fin_nums = fin_nums;
             aff(tot_aff).amp_resp(n_amps).fr_per= fr_per;
             aff(tot_aff).amp_resp(n_amps).pr_per = pr_per;
             base_frs(n_amps,:) = base_fr;
                   
        %end
    
        %overall_base = mean(base_frs,1);
        %for n_amps = 1:length(aff_file_names)
             
             aff(tot_aff).amp_resp(n_amps).base_fr = base_fr;%overall_base;
             aff(tot_aff).amp_resp(n_amps).aff_num = aff_num;
             
        end
            
           tot_aff = tot_aff + 1;
    end
end

all_sing_slope = [sing_resp([1 3:6]).fr_slope_per_points];

%% PLOT PFR AND SLOPES
%==== Supplemental Figure 2 B/ Figure 4 D (PFRS and slope plots) =====
n_fig = 4;
col_blind_maps= load('colorblind_colormap.mat');

I_cols=repmat(linspace(.65,0,5)',[1 3])%col_blind_maps.colorblind([1 5 6 8 9 10],:)
all_af = [1 4:6]; %2/3 were eliminated for poor sortability post hoc
aff_cnt = 1;
for n = 1:length(all_af)

    af_n = all_af(n)% 4:6];% 4:6]
    [srted,srt_ord]=sort([aff(af_n).amp_resp.I]);
    for n_I = 1:length(srt_ord)
        tmp_resp = aff(af_n).amp_resp(srt_ord(n_I));
        plot_pfr_slopes(af_n,tmp_resp,n_fig,I_cols(n_I,:),[2,5],aff_cnt,[]) %Made general function - below for plotting these
    end
    aff_cnt = aff_cnt + 1;
end
%%
all_amp_slope = [];

for n = 1:length(all_af)
    n_aff = all_af(n)
    all_amp_slope =[all_amp_slope  aff(n_aff).amp_resp(:).fr_slope_per_points];
end

edges = [-1:.1:1];

% figure(9); h= histogram([all_amp_slope all_sing_slope],edges); box off;
% xlabel('Slope (sps/pps)'); ylabel('Number of Sampled Firing Rates')
% set(gca,'fontsize',15)

%% LOADING IN SIMULATED DATA

%COMPARE WITH SIMULATION
load(fullfile(data_dir,'sim_slopes_per_S_5_15_23'))% Where perS_dat comes from
tmp = load(fullfile(data_dir,perS_dat(1).name))

for n_S = 1:7
    tmp = strsplit(perS_dat(n_S).name,'_');
    S_val_sim(n_S) = str2num(tmp{3}(2:end));
end
%Group by baseline Firing rate:

clear sing_resp_S_bin amp_resp_S_bin
for n_sing = [1 3:length(sing_resp)]
    [x,rel_S]=min(abs(S_val_sim-sing_resp(n_sing).base_fr(1)));
    sing_resp_S_bin(n_sing) = rel_S;
end

for n_ampaff = [1 4:length(aff)]
    base_per = vertcat(aff(n_ampaff).amp_resp.base_fr);
    avg_base= mean(base_per(:,1));
    [x,rel_S]=min(abs(S_val_sim-avg_base));
    amp_resp_S_bin(n_ampaff) = rel_S;
end
%% Slope histogram plots with statistics
% Slope Comparison
for n_S = 1:7
    perS_slope(n_S).slopes= [sing_resp(sing_resp_S_bin==n_S).fr_slope_per_points];
     perS_slope(n_S).slopes2 = [];
    for n_amp_rec = find(amp_resp_S_bin==n_S)
        n_amp_rec
        perS_slope(n_S).slopes2 = [perS_slope(n_S).slopes2 aff(n_amp_rec).amp_resp(:).fr_slope_per_points];
    end
end

%================ Supplemental Figure 3 C ==========================
figure(9);
plt_cnt =1;

hist_range = -1:0.05:1;
for n_S = 1:7
    all_slope_S = [perS_slope(n_S).slopes2 perS_slope(n_S).slopes];
    if ~isempty(all_slope_S)
        tmp = squeeze(slopes_per_S(n_S,:,:));
        sim_per_s = tmp(:);
        [p1 x1]= hist(all_slope_S,hist_range ); hold on;%,5:1:15); hold on;
        [p2 x2] = hist(sim_per_s,hist_range );%,5:1:15)
        subplot(1,4,plt_cnt);
%         plot(x1,p1/sum(p1),'k','linewidth',2); hold on;
%         plot(x2,p2/sum(p2),'r','linewidth',2)
        histogram(all_slope_S,hist_range, 'Normalization', 'probability','FaceColor','k'); hold on;
        histogram(sim_per_s,hist_range, 'Normalization', 'probability','FaceColor','r'); 
        [h_smir p_smir] = kstest2(sim_per_s,all_slope_S');
        WS_Dist_cur =Wasserstein_Dist(sim_per_s,all_slope_S');
        p_smir_per_S(n_S) = p_smir;
        WS_Dist_cur_per_S(n_S) = WS_Dist_cur;
        legend(sprintf('Experimental (n=%d)',length(all_slope_S)), ...
            sprintf('Simulated (n=%d)',length(sim_per_s)));%,'Simulated with N(0,0.2) Noise');
        box off;
        set(gca,'fontsize',15);
        title(sprintf('S %0.2f, p = %0.3f,W=%0.3f', S_val_sim(n_S), p_smir, WS_Dist_cur))

        plt_cnt =plt_cnt+1;
    end
    perm_counter(n_S) = length( all_slope_S); %How many slopes belonged to each S value
end

%OVERALL:

expt_dist = [all_amp_slope all_sing_slope];
sim_dist = slopes_per_S(:);
[h p_ks2_overall] = kstest2(expt_dist,sim_dist)
WS_Dist_overall = Wasserstein_Dist(expt_dist',sim_dist')

[p1 x1]= hist(expt_dist,hist_range ); hold on;%,5:1:15); hold on;
[p2 x2] = hist(sim_dist,hist_range );%,5:1:15)

%==================== Figure 4 E ======================%
figure(10); 
histogram(expt_dist,hist_range, 'Normalization', 'probability','FaceColor','k'); hold on;
histogram(sim_dist, hist_range, 'Normalization', 'probability','FaceColor','r');
legend(sprintf('Experimental (n=%d)',length(expt_dist)), ...
    sprintf('Simulated (n=%d)',length(sim_dist)));%,'Simulated with N(0,0.2) Noise');
set(gca,'fontsize',15); xlabel('Slope (sps/pps)'); ylabel(''); box off
 title(sprintf('S %0.2f, p = %0.3f,W=%0.3f', S_val_sim(n_S),p_ks2_overall, WS_Dist_overall))
%%
%Instead of perming slopes
%all_slopes_all_categories = [perS_slope.slopes perS_slope.slopes2];
good_sing_affs = [1 3:6];
good_multiI_affs = [1 4:6];

%Combining all prs, all frs into one set of two vectors from the good real
%experimental recordings
aff_ref_cnt =1;
all_prs =[]; all_frs =[]; aff_order = []; aff_S = [];
for n_sing_aff = 1:length(sing_resp)
    if ismember(n_sing_aff,good_sing_affs)
        %sing_resp(n_sing_aff)
        all_prs = [all_prs sing_resp(n_sing_aff).pr_per];
        all_frs = [all_frs sing_resp(n_sing_aff).fr_per];
        aff_order = [aff_order aff_ref_cnt*ones(size(sing_resp(n_sing_aff).fr_per))];
        [a S_idx] = min(abs(S_val_sim - sing_resp(n_sing_aff).fr_per(1)));
        aff_S = [aff_S S_idx*ones(size(sing_resp(n_sing_aff).fr_per))];
        aff_ref_cnt =aff_ref_cnt + 1;
    end
end
%Adding in the per amplitude trials:
for n_aff = 1:length(aff)
    if ismember(n_aff,good_multiI_affs) % used afferents after eliminating poor recording/sorting
        base_frs=vertcat(aff(n_aff).amp_resp(:).base_fr);
        avg_base_fr = mean(base_frs(:,1));
        [a S_idx] = min(abs(S_val_sim - avg_base_fr ));
        for n_a = 1:length(aff(n_aff).amp_resp)
            all_frs = [all_frs aff(n_aff).amp_resp(n_a).fr_per];
            all_prs = [all_prs aff(n_aff).amp_resp(n_a).pr_per];
            aff_order = [aff_order aff_ref_cnt*ones(size(aff(n_aff).amp_resp(n_a).fr_per))];
            
           aff_S = [aff_S S_idx*ones(size(aff(n_aff).amp_resp(n_a).fr_per))];
            aff_ref_cnt =aff_ref_cnt + 1;
        end
    end
end
%% Permuting the matching pr/fr combinations and getting slopes to compare with the original slopes from the data
%23 different recordings, 143 fr values
%perm_counter - # slopes per S values for comparison per S
slope_calc = @(fr_per,pr_per) diff(fr_per)./diff(pr_per);
clear perm_slopes
n_cnt =1;
n_perms = 5000;
for n_perm =1:n_perms
    perm_slopes(3).slopes = [];
    perm_slopes(4).slopes = [];
    perm_slopes(6).slopes = [];
    cur_frs = all_frs(randperm(length(all_frs)));
    for n_trace = 1:length(unique(aff_order))
        rel_idx = find(aff_order == n_trace);
        perm_slopes(unique(aff_S(rel_idx))).slopes = [perm_slopes(unique(aff_S(rel_idx))).slopes ...
            slope_calc(cur_frs(rel_idx),all_prs(rel_idx))];
    end
    for n_S = [3 4 6] % based on S_val_sim - 13,28, 5 sps
        tmp = squeeze(slopes_per_S(n_S,:,:));
        sim_per_s = tmp(:);
        %Stat Tests:
        [h_smir p_smir] = kstest2(sim_per_s,perm_slopes(n_S).slopes');
        WS_Dist_cur =Wasserstein_Dist(sim_per_s,perm_slopes(n_S).slopes');
        perm_smir_p(n_S,n_perm) = p_smir;
        perm_WS_dist(n_S,n_perm) = WS_Dist_cur;
        n_cnt = (n_cnt+perm_counter(n_S));

    end
    %Overall:
    [h_smir p_smir] = kstest2( [slopes_per_S(:)],[perm_slopes(:).slopes]');
    WS_Dist_cur =Wasserstein_Dist([slopes_per_S(:)],[perm_slopes(:).slopes]');
    overall_psmir_per(n_perm) = p_smir;
    overall_WS_per(n_perm) = WS_Dist_cur;

end

%% PERM TEST ALL S values: (across all cases combined)
mean(overall_psmir_per);
mean(overall_WS_per);
figure(6);
subplot(2,1,1); histogram(overall_psmir_per); hold on;
p_chance_ks =sum(overall_psmir_per > p_ks2_overall)/length(overall_psmir_per);
plot([p_ks2_overall p_ks2_overall],[0 3000],'r','linewidth',2); 
title(sprintf('KS mean = %0.2e,P =%0.2f %', ...
        mean(overall_psmir_per),p_chance_ks)); box off;
subplot(2,1,2); histogram(overall_WS_per); hold on;
prob_chance_WS(n_S) =sum(overall_WS_per < WS_Dist_overall)/n_perms;
plot([WS_Dist_overall WS_Dist_overall],[0 500],'r','linewidth',2);
title(sprintf('WS mean = %0.2e,P =%0.2f %', ...
        mean(WS_Dist_overall),prob_chance_WS)); box off;

%=========================================================================
%=========================================================================
%% Slope vs. Pulse Rate stats:
%simulated slope data pulled from I_S_fit_2_21_23.m
%Newest analysis to look at a slope v. pulse rate function and if similar in
%the data and in the simulation
load(fullfile(data_dir,'sim_slopes_per_S_5_15_23'))%
sim_dat = load(fullfile(data_dir,perS_dat(1).name));
sim_info =load(fullfile(data_dir,'sim_slopes_per_S_5_15_23'));%'slopes_per_S_7_03_23'));%'
sampled_Is =  sim_dat.per_S.I(sim_info.I_range)*-20;
slope_cnt_pr = sim_info.pr_range(1:end-1)+diff(sim_info.pr_range(1:2))/2;
s_cols =winter(7);
i_cols = parula(9);
[a re_ord]=sort(sim_info.S_ord);

%% Lower vs. higher PR under diff S and diff I:
%============= Supplemental Figure 3 D =============================
thresh_pr = 150;
split_prs = ((slope_cnt_pr*2) < thresh_pr);
%Spont Rate, prs, I levels
rel_simulated_lowpr = squeeze(sim_info.slopes_per_S(re_ord(2:5),split_prs,1:size(sim_info.slopes_per_S,3)));
rel_simulated_highpr = squeeze(sim_info.slopes_per_S(re_ord(2:5),~split_prs,1:size(sim_info.slopes_per_S,3)));
figure(200)
for n_rate = 1:size(rel_simulated_lowpr,1)
    subplot(4,1,n_rate);
    bar(1:size(rel_simulated_lowpr,3),[squeeze(mean(rel_simulated_lowpr(n_rate,:,:),2))...
        squeeze(mean(rel_simulated_highpr(n_rate,:,:),2))]); hold on;
     errorbar([1:size(rel_simulated_lowpr,3)]-.15,squeeze(mean(rel_simulated_lowpr(n_rate,:,:),2)),...
        squeeze(std(rel_simulated_lowpr(n_rate,:,:),[],2))/sqrt(size(rel_simulated_lowpr,2)),'k.');
      errorbar([1:size(rel_simulated_highpr,3)]+.15,squeeze(mean(rel_simulated_highpr(n_rate,:,:),2)),...
        squeeze(std(rel_simulated_highpr(n_rate,:,:),[],2))/sqrt(size(rel_simulated_highpr,2)),'k.');
    title(sprintf('S %0.2f',a(n_rate+1)));
    ylim([-1 1.2])
end
xticklabels(sampled_Is)
%% Do per I statistics/power testing:-----
for n_slevel = 1:size(sim_info.slopes_per_S,1)
        for n_Is = 1:size(sim_info.slopes_per_S,3)
            
            pop_1 = squeeze(sim_info.slopes_per_S(n_slevel,(split_prs),(n_Is)));
            pop_2 =  squeeze(sim_info.slopes_per_S(n_slevel,~(split_prs),(n_Is)));

            mu_2 = mean(pop_2);
            std_2= std(pop_2);
            mu_1 = mean(pop_1);
            std_1 = std(pop_1);
            n_per_test = min(length(pop_1),length(pop_2));

            if (std_1 ~=0) & (std_2~=0)
                nout = sampsizepwr('t',[mu_1 std_1],mu_2,0.95,[],'Tail','both');
                power = sampsizepwr('t',[mu_1 std_1],mu_2,[],n_per_test,'Tail','both');
            else
                nout = nan;power = nan;
            end

            [h,p,~,stats] = ttest2(pop_1,pop_2,"Tail","both");
            disp(sprintf('N needed %d, power %0.3f',nout,power));
            stat_power_per_sim(n_slevel).I_level(n_Is).pow = power;
            stat_power_per_sim(n_slevel).I_level(n_Is).n_samples = nout;
            stat_power_per_sim(n_slevel).I_level(n_Is).stats = stats;
            stat_power_per_sim(n_slevel).I_level(n_Is).p_val = p;

        end
end

%% Experimental slope grouping for same analysis
% sing_resp_S_bin
% amp_resp_S_bin

%Ignoring the 5 case because don't have at multiple amplitudes? so only 4
%afferents
idx_s_sing = unique([sing_resp_S_bin amp_resp_S_bin]);
idx_s_sing = idx_s_sing(2:end);
for n_ss = 1:length(idx_s_sing)
    rel_multi_I = find(amp_resp_S_bin == idx_s_sing(n_ss));
    sing_I = find(sing_resp_S_bin == idx_s_sing(n_ss)); %only have for single I so maybe not so helpful? this is why the 5 sps not in most of analyses
    S_exps(n_ss).base_fr_grp = S_val_sim(idx_s_sing(n_ss));
    for n = 1:length(rel_multi_I)
        %organize into same prs:
        all_prs =  unique([aff(rel_multi_I(n)).amp_resp.pr_centers]);
        resp_w_pr = nan(length(aff(rel_multi_I(n)).amp_resp),length(all_prs));
        S_exps(n_ss).aff(n).slopes = resp_w_pr;
        for nI = 1:length(aff(rel_multi_I(n)).amp_resp)
            S_exps(n_ss).aff(n).slopes(nI,ismember(all_prs,aff(rel_multi_I(n)).amp_resp(nI).pr_centers)) ...
                = aff(rel_multi_I(n)).amp_resp(nI).fr_slope_per_points;
            S_exps(n_ss).aff(n).I(nI) = aff(rel_multi_I(n)).amp_resp(nI).I;
            S_exps(n_ss).aff(n).prs = all_prs;%aff(rel_multi_I(n)).amp_resp(nI).pr_centers;
        end
    end

    S_exps(n_ss).grp = idx_s_sing(n_ss);
end

%%
S_exps = S_exps(1:2);
%Combine across Afferents:
for n = 1:length(S_exps)
    sampled_Is =unique([S_exps(n).aff.I]);
    sample_prs= unique([S_exps(n).aff.prs]);
    af_I_pr = nan(length(S_exps(n).aff),length(sampled_Is),length(sample_prs));
    for n_aff = 1:length(S_exps(n).aff)
        for nIs = 1:length(S_exps(n).aff(n_aff).I)
            af_I_pr(n_aff,sampled_Is == S_exps(n).aff(n_aff).I(nIs),...
                ismember(sample_prs,S_exps(n).aff(n_aff).prs)) = S_exps(n).aff(n_aff).slopes(nIs,:);
        end
    end
    S_exps(n).combined = af_I_pr;
    S_exps(n).comb_prs = sample_prs;
    S_exps(n).comb_I = sampled_Is;
end %using nans should be able to correctly sort and account for not same Is or same prs sampled

%% Same low/high pr analysis for Experimental data:
%=================== Supplemental Figure 2 C ===========================
figure(201);

aff_cnt = 1;

all_Is = unique([S_exps(1).comb_I S_exps(2).comb_I]);

%only place at relevant Is then
for n = 1:2
    all_prs = S_exps(n).comb_prs;
    split_prs = all_prs< thresh_pr;
    

    %per_I_resp = [];
    for n_aff = 1:size(S_exps(n).combined,1)

        all_resp_low =nanmean((S_exps(n).combined(n_aff,:,(split_prs))),3);
        all_resp_high =nanmean((S_exps(n).combined(n_aff,:,~(split_prs))),3);
        all_resp_low_std =nanstd((S_exps(n).combined(n_aff,:,(split_prs))),[],3);
        all_resp_high_std =nanstd((S_exps(n).combined(n_aff,:,~(split_prs))),[],3);

        %For power testing:
        %T-test but accompanied with a power statistics check of above 80%
        % spearman rank (rho) test
        n_per_test= min(max(sum(~isnan(squeeze(S_exps(n).combined(n_aff,:,~(split_prs)))),2)),...
        max(sum(~isnan(squeeze(S_exps(n).combined(n_aff,:,(split_prs)))),2)));


%============================================================%
% Check on power and do t-test:
        check_tests = find(~isnan(all_resp_low));
        for n_Is = 1:length(check_tests)
            mu_2 = all_resp_high(check_tests(n_Is));
            std_2= all_resp_high_std(check_tests(n_Is ));
            mu_1 = all_resp_low(check_tests(n_Is ));
            std_1 = all_resp_low_std(check_tests(n_Is ));

            pop_1 = squeeze(S_exps(n).combined(n_aff,check_tests(n_Is),(split_prs)));
            pop_1 = pop_1(~isnan(pop_1));
            pop_2 = squeeze(S_exps(n).combined(n_aff,check_tests(n_Is),~(split_prs)));
            pop_2 = pop_2(~isnan(pop_2));

                nout = sampsizepwr('t',[mu_1 std_1],mu_2,0.95,[],'Tail','both');
                power = sampsizepwr('t',[mu_1 std_1],mu_2,[],n_per_test,'Tail','both');

            [h,p,~,stats] = ttest2(pop_1,pop_2,"Tail","both");
            disp(sprintf('N needed %d, power %0.3f',nout,power));
            stat_power_per(aff_cnt).I_level(n_Is).pow = power;
            stat_power_per(aff_cnt).I_level(n_Is).n_samples = nout;
            stat_power_per(aff_cnt).I_level(n_Is).stats = stats;
            stat_power_per(aff_cnt).I_level(n_Is).p_val = p;

            %percent high pr that are less than 0 with increasing I
            stat_power_per(aff_cnt).perc_neg(n_Is,:)  = [sum(pop_1 <= 0)/length(pop_1) sum(pop_2 <= 0)/length(pop_2)];


        end
%============================================================%

        subplot(4,1,aff_cnt)
        xticks([1:length(all_Is)]); hold on;
        %per_I_resp_lowpr = nanmean(all_resp_low(n_aff,:),1)';
        %per_I_resp_highpr = nanmean(all_resp_high(n_aff,:),1)';
        %end
        rel_x_val = find(ismember(all_Is,S_exps(n).comb_I));
        bar(rel_x_val,[all_resp_low' all_resp_high']); hold on;
        errorbar([rel_x_val]+.15, all_resp_high,...
            all_resp_high_std/sqrt(sum(~split_prs)),'k.');
        errorbar([rel_x_val]-.15,all_resp_low,...
            all_resp_low_std/sqrt(sum(split_prs)),'k.');
        %nanstd((S_exps(n).combined(:,:,(split_prs))),[],3)
        xticklabels(all_Is);
        ylim([-1 1.2]); xlim([0 length(all_Is)+1])
        title(sprintf('Aff %d %0.1f sps',aff_cnt,S_exps(n).base_fr_grp));
        aff_cnt = aff_cnt + 1;

    end
end

%% Cross all afferents but high amplitude:

%Ignoring the 5 case because don't have at multiple amplitudes? so only 4
%afferents
idx_s_sing = unique([sing_resp_S_bin amp_resp_S_bin]);
idx_s_sing = idx_s_sing(2:end);
S_max(n_ss).slopes_low = [];
S_max(n_ss).slopes_high = [];
for n_ss = 1:length(idx_s_sing)
    rel_multi_I = find(amp_resp_S_bin == idx_s_sing(n_ss));
    sing_I = find(sing_resp_S_bin == idx_s_sing(n_ss)); %only have for single I so maybe not so helpful? this is why the 5 sps not in most of analyses
    S_max(n_ss).base_fr_grp = S_val_sim(idx_s_sing(n_ss));
    for n = 1:length(rel_multi_I)
        %organize into same prs:
        [a idx_imax]=max([aff(rel_multi_I(n)).amp_resp.I]);
        all_prs =  aff(rel_multi_I(n)).amp_resp(idx_imax).pr_centers;
        split_pr = all_prs <thresh_pr;
        %resp_w_pr = nan(length(aff(rel_multi_I(n)).amp_resp),length(all_prs));
        %S_exps(n_ss).aff(n).slopes = resp_w_pr;
        %for nI = 1:length(aff(rel_multi_I(n)).amp_resp)
            
            S_max(n_ss).slopes_low = [S_max(n_ss).slopes_low ...
            aff(rel_multi_I(n)).amp_resp(idx_imax).fr_slope_per_points(split_pr)];

            S_max(n_ss).slopes_high = [S_max(n_ss).slopes_high ...
            aff(rel_multi_I(n)).amp_resp(idx_imax).fr_slope_per_points(~split_pr)];

            S_max(n_ss).I_per(n) = aff(rel_multi_I(n)).amp_resp(idx_imax).I;
            %S_exps(n_ss).aff(n).prs = all_prs;%aff(rel_multi_I(n)).amp_resp(nI).pr_centers;
        %end
    end
    %Can include here as well if want to (the single high amplitude only- but don't have details on I):
      n_sofar= length(rel_multi_I);%length(S_exps(n_ss).aff);
    for n_s = 1:length(sing_I)

       S_max(n_ss).slopes_low = [S_max(n_ss).slopes_low ...
            sing_resp(sing_I(n_s)).fr_slope_per_points(sing_resp(sing_I(n_s)).pr_centers < thresh_pr)];
       S_max(n_ss).slopes_high = [S_max(n_ss).slopes_high ...
            sing_resp(sing_I(n_s)).fr_slope_per_points(sing_resp(sing_I(n_s)).pr_centers >= thresh_pr)];

    end

end
%% High-low split per S Value
%%Plotting across group:
avg_l =[mean(S_max(1).slopes_low) mean(S_max(2).slopes_low) mean(S_max(3).slopes_low)]
std_l =[std(S_max(1).slopes_low) std(S_max(2).slopes_low) std(S_max(3).slopes_low)]./...
    sqrt([length(S_max(1).slopes_low), length(S_max(2).slopes_low) length(S_max(3).slopes_low)]);
avg_h = [mean(S_max(1).slopes_high) mean(S_max(2).slopes_high) mean(S_max(3).slopes_high)]
std_h = [std(S_max(1).slopes_high) std(S_max(2).slopes_high) std(S_max(3).slopes_high)]./...
    sqrt([length(S_max(1).slopes_high), length(S_max(2).slopes_high) length(S_max(3).slopes_high)]);
% figure(1);bar([1:3],[avg_l; avg_h]); hold on;
% errorbar([-.15+[1 2 3] .15+[1 2 3]],[avg_l avg_h],[std_l std_h],'k.');
%% ========================= Figure 4 F ================================
figure(2);
bar([1 2],[mean([S_max.slopes_low]) mean([S_max.slopes_high])]); hold on;
errorbar([1 2],[mean([S_max.slopes_low]) mean([S_max.slopes_high])],...
    [std([S_max.slopes_low])/sqrt(length([S_max.slopes_low])) ...
    std([S_max.slopes_high])/sqrt(length([S_max.slopes_high]))],'k.');
plot(.25*(.5-rand(length([S_max.slopes_low]),1))+1,[S_max.slopes_low],'.');
plot(.25*(.5-rand(length([S_max.slopes_high]),1))+2,[S_max.slopes_high],'.');

nout = sampsizepwr('t',[mean([S_max.slopes_low]) std([S_max.slopes_low])],mean([S_max.slopes_high]),0.95,[],'Tail','both');
power = sampsizepwr('t',[mean([S_max.slopes_low]) std([S_max.slopes_low])],mean([S_max.slopes_high]),[],min(length([S_max.slopes_high]),length([S_max.slopes_low])),'Tail','both');
[h,p,~,stats] = ttest2([S_max.slopes_low],[S_max.slopes_high],"Tail","both")
xticklabels({'Low PR','High PR'}); ylabel('Slope (sps/pps)')
%%
 figure(3);
hl = histogram([S_max.slopes_low],[-.5:.1:1.1]); hold on;
pl= histcounts([S_max.slopes_low],[-.5:.1:1.1],'normalization','pdf');
hh = histogram([S_max.slopes_high],[-.5:.1:1.1]);
ph= histcounts([S_max.slopes_high],[-.5:.1:1.1],'normalization','pdf');

binCenters_l = hl.BinEdges + (hl.BinWidth/2);
binCenters_h = hh.BinEdges + (hh.BinWidth/2);
%% ======================= Figure 4 F =============================
figure(4);
pl= histogram([S_max.slopes_low],[-.5:.1:1.1],'normalization','pdf'); hold on;
ph= histogram([S_max.slopes_high],[-.5:.1:1.1],'normalization','pdf');
xlabel('(sps/pps)'); ylabel('Occurance')

Wasserstein_Dist([S_max.slopes_low]',[S_max.slopes_high])';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTIONS FOR ABOVE

function [fin_nums base_fr] = data_split(data_per_aff,plot_it,cur_neur,aff_num,n_neurons,afferent_info,n_plot)
pulse_timing = find(data_per_aff.stim);
pulse_block_div = find(diff(pulse_timing) > 250);
pulse_block_start = [pulse_timing(1); pulse_timing(pulse_block_div+1)];
pulse_block_end = [pulse_timing(pulse_block_div); pulse_timing(end)];

single_spk = (pulse_block_start == pulse_block_end);
pulse_block_start(single_spk) = [];
pulse_block_end(single_spk) = [];
if plot_it
    %For each amplitude per neuron
    %subplot(3,1,1)

    plot(data_per_aff.stim); hold on;
    plot(pulse_block_start,data_per_aff.stim(pulse_block_start),'r.'); hold on;
    plot(pulse_block_end,data_per_aff.stim(pulse_block_end),'b.');
    xlabel('Time (ms)')
    %pause; %cla %%%%Note afferent 3 had weird block spacing
end

%Get data per block
clear base_firing_btw_block
min_inter_wind_length = 300;%before 1000 ms
for n_p = 1:length(pulse_block_start)
    p_rate_per_block(n_p) = sum((pulse_timing >= pulse_block_start(n_p)) & ...
        (pulse_timing <= pulse_block_end(n_p)))/(1e-3*(pulse_block_end(n_p) - pulse_block_start(n_p)));
    firing_rate_per_block(n_p) = sum((find(data_per_aff.ua) >= pulse_block_start(n_p)) & ...
        (find(data_per_aff.ua) <= pulse_block_end(n_p)))/(1e-3*(pulse_block_end(n_p) - pulse_block_start(n_p)));
   
    %Extra information for window testing:
    n_pulse(n_p) = sum((pulse_timing >= pulse_block_start(n_p)) & ...
        (pulse_timing <= pulse_block_end(n_p)));
    n_aps(n_p) =  sum((find(data_per_aff.ua) >= pulse_block_start(n_p)) & ...
        (find(data_per_aff.ua) <= pulse_block_end(n_p)));
    

    %Estimate base firing rate during the trial:
    %Only use up to a 1000 ms before data to avoid too close to last trial:
    %Don't use data within 100 ms of the last trial end
    %max_wind = 5000; %ms
    dist_last = 50; %ms
    if n_p ==1
        if pulse_block_start(n_p) < min_inter_wind_length %ms
            % Have no data like this
            warning(sprintf('Aff %s block %s issue',num2str(aff_num),num2str(n_p)));
            base_firing_btw_block(n_p) = nan;
        else
            %1000 ms max
%             base_firing_btw_block(n_p) = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
%               (find(data_per_aff.ua) >= (pulse_block_start(n_p) - max_wind)))/(1e-3*(pulse_block_start(n_p)));
            base_firing_btw_block(n_p) = sum(find(data_per_aff.ua) < pulse_block_start(n_p))/(1e-3*(pulse_block_start(n_p)));
        end
    else
        if (pulse_block_start(n_p) - pulse_block_end(n_p-1)) < min_inter_wind_length
             warning(sprintf('Aff %s block %s issue',num2str(aff_num),num2str(n_p)));
            last_idx = max(find(~isnan(base_firing_btw_block(1:(n_p-1)))));
            cur_idx  = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
                (find(data_per_aff.ua) > ...
               (pulse_block_end(n_p-1)+ dist_last))) ...
                /(1e-3*(pulse_block_start(n_p) - pulse_block_end(n_p-1)));
            base_firing_btw_block(n_p) = mean([base_firing_btw_block(last_idx),cur_idx]);%nan;
            % Consider averaging in with past data instead of this (try
            % this first)
        else
%             base_firing_btw_block(n_p) = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
%                 (find(data_per_aff.ua) > ...
%                 max((pulse_block_end(n_p-1)+ dist_last),pulse_block_start(n_p) - max_wind))) ...
%                 /(1e-3*(pulse_block_start(n_p) - pulse_block_end(n_p-1)));
 base_firing_btw_block(n_p) = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
                (find(data_per_aff.ua) > ...
               (pulse_block_end(n_p-1)+ dist_last))) ...
                /(1e-3*(pulse_block_start(n_p) - pulse_block_end(n_p-1)));
        end
    end
end
base_fr = [mean(base_firing_btw_block) std(base_firing_btw_block)];
dfr = firing_rate_per_block - mean(base_firing_btw_block);

 [coeff] =best_fit_line(p_rate_per_block,dfr);
 
 fin_nums.slope = coeff.Coefficients.Estimate;
% Print out info:
verbose = 0;
if verbose
   disp(sprintf('Aff %s ',num2str(aff_num)));
    clear len_pre_wind
    len_pr_wind = pulse_block_end -  pulse_block_start;
    len_pre_wind(1) = pulse_block_start(1);
    len_pre_wind(2:(length(pulse_block_start))) = pulse_block_start(2:end) - pulse_block_end(1:(end-1));
    
    disp('Wind lengths PR/btw (s); N P AP in PR wind; PR FR in wind')
    disp([round(len_pr_wind/1e3,3), round(len_pre_wind/1e3,3)',n_pulse',n_aps',round(p_rate_per_block,3)',round(firing_rate_per_block,3)'])
end
 
%Combine across same pulse rate: - make sure only getting one pr per pr
%group
unique_ps = unique(round(p_rate_per_block));
un_ps = [];
for n_u_ps = 1:length(unique_ps)
    if ~(sum(abs(unique_ps(n_u_ps) - round(unique_ps)) < 3) > 1)
       
        un_ps = [un_ps unique_ps(n_u_ps)];  
    end
end
 
%un_ps;
unique_ps = un_ps; un_ps = [];
for n_unique_ps = 1:length(unique_ps)
    pr_n(n_unique_ps).idxs = find(abs(unique_ps(n_unique_ps) - round(p_rate_per_block)) < 3);
    fin_nums.pr(n_unique_ps,:) = [mean(round(p_rate_per_block(pr_n(n_unique_ps).idxs))) ...
        std(round(p_rate_per_block(pr_n(n_unique_ps).idxs)))];
    fin_nums.fr(n_unique_ps,:) = [mean(round(firing_rate_per_block(pr_n(n_unique_ps).idxs))) ...
        std(round(firing_rate_per_block(pr_n(n_unique_ps).idxs)))];
    fin_nums.dfr(n_unique_ps,:) = [mean(round(dfr(pr_n(n_unique_ps).idxs))) ...
        std(round(dfr(pr_n(n_unique_ps).idxs)))];
end

    % 
    % figure(n_plot); subplot(2,3,cur_neur);%subplot(n_neurons,1,cur_neur);
    % errorbar([0; fin_nums.pr(:,1)],[base_fr(:,1); fin_nums.fr(:,1)],...
    %     [base_fr(:,2); fin_nums.fr(:,2)],'.-');hold on; 
    % 
    % xlabel('Pulse Rate'); ylabel('Firing Rate'); box off;
    % title(sprintf('Afferent %s: CV*: %s',num2str(aff_num), num2str(round(afferent_info(aff_num,5),2))));
per_trial.n_frs = firing_rate_per_block;
per_trial.prs = p_rate_per_block;
per_trial.base_frs= base_firing_btw_block;
end


%%% ANALYSES MITCHELL ET AL. DATA!
function [fin_nums base_fr block_dat] = data_t_fr(data_per_aff,plot_it,cur_neur,aff_num,n_neurons,afferent_info,n_plot)
pulse_timing = find(data_per_aff.stim);
pulse_block_div = find(diff(pulse_timing) > 250);
pulse_block_start = [pulse_timing(1); pulse_timing(pulse_block_div+1)];
pulse_block_end = [pulse_timing(pulse_block_div); pulse_timing(end)];

single_spk = (pulse_block_start == pulse_block_end);
pulse_block_start(single_spk) = [];
pulse_block_end(single_spk) = [];
if plot_it
    figure(1); %For each amplitude per neuron
    subplot(3,1,1)
    plot(data_per_aff.stim); hold on;
    plot(pulse_block_start,data_per_aff.stim(pulse_block_start),'r.'); hold on;
    plot(pulse_block_end,data_per_aff.stim(pulse_block_end),'b.');
    xlabel('Time (ms)')
    %pause; %cla %%%%Note afferent 3 had weird block spacing
end

%Get data per block
min_inter_wind_length = 300;%500;%before 1000 ms
for n_p = 1:length(pulse_block_start)
    
    p_rate_per_block(n_p) = sum((pulse_timing >= pulse_block_start(n_p)) & ...
        (pulse_timing <= pulse_block_end(n_p)))/(1e-3*(pulse_block_end(n_p) - pulse_block_start(n_p)));
    firing_rate_per_block(n_p) = sum((find(data_per_aff.ua) >= pulse_block_start(n_p)) & ...
        (find(data_per_aff.ua) <= pulse_block_end(n_p)))/(1e-3*(pulse_block_end(n_p) - pulse_block_start(n_p)));
    %Estimate base firing rate during the trial:
    if n_p ==1
        if pulse_block_start(n_p) < min_inter_wind_length %ms
            base_firing_btw_block(n_p) = nan;
        else
            base_firing_btw_block(n_p) = sum(find(data_per_aff.ua) < pulse_block_start(n_p))/(1e-3*(pulse_block_start(n_p)));
        end
    else
        if (pulse_block_start(n_p) - pulse_block_end(n_p-1)) < min_inter_wind_length
            last_idx = max(find(~isnan(base_firing_btw_block(1:(n_p-1)))));
            base_firing_btw_block(n_p) = base_firing_btw_block(last_idx);%nan;
        else
            base_firing_btw_block(n_p) = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
                (find(data_per_aff.ua) > pulse_block_end(n_p-1)))/(1e-3*(pulse_block_start(n_p) - pulse_block_end(n_p-1)));
        end
    end
end
%Estimate mean and std across trial
base_fr = [nanmean(base_firing_btw_block) nanstd(base_firing_btw_block)];

block_dat.pr = p_rate_per_block;
block_dat.fr = firing_rate_per_block;
block_dat.base_fr = base_firing_btw_block;
sub_base = block_dat.base_fr;
%sub_base(isnan(block_dat.base_fr)) = mean(sub_base(~isnan(block_dat.base_fr)));
dfr = block_dat.fr - sub_base;


%% Fit a slope to increase with each current:
 [coeff] =best_fit_line(block_dat.pr,dfr);
 
% %  figure(1); plot(block_dat.pr,dfr,'k.'); hold on;
% %  plot([0:300],coeff.Coefficients.Estimate*[0:300],'r')
 
 fin_nums.m_per_amp = coeff.Coefficients.Estimate;
%%
if plot_it
%figure(2);
subplot(3,1,2);%subplot(n_neurons,1,cur_neur);
plot(1:length(pulse_block_start),p_rate_per_block,'.'); hold on;
xlabel('Block Num'); ylabel('Pulse Rate');
end

%Combine across same pulse rate:
unique_ps = unique(round(p_rate_per_block));


for n_unique_ps = 1:length(unique_ps)
    pr_n(n_unique_ps).idxs = find(abs(unique_ps(n_unique_ps) - round(p_rate_per_block)) < 2);
    fin_nums.pr_n(n_unique_ps,:) = [mean(round(p_rate_per_block(pr_n(n_unique_ps).idxs))) ...
        std(round(p_rate_per_block(pr_n(n_unique_ps).idxs)))];
    fin_nums.fr_n(n_unique_ps,:) = [mean(round(firing_rate_per_block(pr_n(n_unique_ps).idxs))) ...
        std(round(firing_rate_per_block(pr_n(n_unique_ps).idxs)))];
    fin_nums.dfr_n(n_unique_ps,:) = [mean(round(dfr(pr_n(n_unique_ps).idxs))) ...
        std(round(dfr(pr_n(n_unique_ps).idxs)))];
end

%if plot_it
    figure(n_plot); subplot(2,3,cur_neur)%subplot(n_neurons,1,cur_neur);
    errorbar([0; fin_nums.pr_n(:,1)],[base_fr(:,1); fin_nums.fr_n(:,1)],...
        [base_fr(:,2); fin_nums.fr_n(:,2)],'k.');
    hold on; plot([0; fin_nums.pr_n(:,1)],[base_fr(:,1); fin_nums.fr_n(:,1)],'k');
    xlabel('Pulse Rate'); ylabel('Firing Rate'); box off;
    title(sprintf('Afferent %s: CV*: %s',num2str(aff_num), num2str(round(afferent_info(aff_num,5),2))));
%end
end



function [coeff] =best_fit_line(prs,dfrs)
coeff = fitlm(prs,dfrs,'Intercept',false);
end

function [I_idx,rms_best, fr_pred_best,mean_std_best] = two_d_rms_eval(S_cur,prs_cur,real_y)
%Function for 2d rms  between full prediction and the input data
%%%%%%%%%%%%%%%%
if (size(real_y,1) < size(real_y,2))
    real_y = real_y';
end
prs_tot = 0:400;
I_range = [0:5:500]%360];%[0:360];
for n_Is = 1:length(I_range)
    I_cur = I_range(n_Is);
     [tot_pred ] = interp_pred_f_5_5_21(I_cur,S_cur,prs_tot);
    %[tot_pred] = interp_pred_fr_v2(I_cur,S_cur,prs_tot);
    
    fr_preds(n_Is,:) = S_cur  + tot_pred';
end

for n_prs = 1:length(prs_cur)
    
    real_x = prs_cur(n_prs);
    
    x_diff = prs_tot - real_x;
    x_diff_repped = repmat(x_diff,[length(I_range) 1]);
    y_diff = fr_preds - real_y(n_prs,1);
    dist_y  =(y_diff).^2;
    %Correct off std in y/fr
    %dist_y  =max(0,dist_y - real_y(n_prs,2)^2);
    q =  sqrt((x_diff_repped).^2 +  dist_y);%+ (y_diff).^2);
    %Subtract out anything within std

    [rms_val_pt_n I_idx] = min(q');%min val per current amplitude
    rms_per_pt(n_prs,:) = rms_val_pt_n;
    
end
shape_err = sum(rms_per_pt,1); %total error per current from best point


% comp_preds = fr_preds(:,find(ismember(prs_tot,prs_cur)));
% for n_preds = 1:size(comp_preds,1)
% [pred_corrs(n_preds,:) lags] = xcorr(real_y',comp_preds(n_preds,:));
% end
%
% normed_xcorr_0 = pred_corrs(:,find(lags == 0));
% normed_xcorr_0 = (1 - normed_xcorr_0./max(normed_xcorr_0));

%= cross_pt_rms;%2*(cross_pt_rms/max(cross_pt_rms));%normed_xcorr_0' +

[rmss idx_fins] = sort(shape_err);

tot_err = rmss + rmss.*.5.*idx_fins./200;%shape_err + I_idx;
%[rms_tmp idx_tmp] = sort(tot_err );
[rms_2 idx_2] = min(tot_err);
I_idx = I_range(idx_fins(idx_2));
fr_pred_best = fr_preds(idx_fins(idx_2),:);

rms_best = rmss(idx_2);%(1);
mean_std_best =[mean(rms_per_pt(:,idx_fins(idx_2))) std(rms_per_pt(:,idx_fins(idx_2)))/size(rms_per_pt,1)]
% % cols = jet(length(75:98))
% % figure(100); 
% % for n= 75:98
% % plot(prs_tot,fr_preds(n,:),'color',cols(n-74,:)); hold on;
% % end
% % errorbar(prs_cur,real_y(:,1),real_y(:,2),'ko'); hold on;
% %  plot(prs_tot,fr_pred_best,'k','linewidth',2); 
end


function [] = plot_pfr_slopes(n_aff,sing_resp,n_fig,line_col,subplot_ns,n_p,base_fr)
%n_aff is used indexs

figure(n_fig);
if isempty(n_p)
    n_p = 1;
    rec_mode = 1;
else
    rec_mode =0;
    n_aff= 1;
end

for n_a = 1:length(n_aff)
    n = n_aff(n_a);

if isempty(base_fr)
    base_fr = sing_resp(n).base_fr;
end
    subplot(subplot_ns(1),subplot_ns(2),n_p);
    errorbar([0 sing_resp(n).fin_nums.pr(:,1)'],[base_fr(:,1) sing_resp(n).fin_nums.fr(:,1)'],...
        [base_fr(:,2) sing_resp(n).fin_nums.fr(:,2)'],'.-','color',line_col,'markersize',12);hold on;
    if n_p == 1
        ylabel('Firing Rate (sps)')
    end
    title(['Afferent ' num2str(sing_resp(n).aff_num)]); box off;
    set(gca,'fontsize',15);xlim([0 350])


    subplot(subplot_ns(1),subplot_ns(2),subplot_ns(2)+n_p);
    if rec_mode 
    plot(sing_resp(n).pr_centers,sing_resp(n).fr_slope_per_points,'.-','markersize',15); hold on;
    else
    plot(sing_resp(n).pr_centers,sing_resp(n).fr_slope_per_points,'.-','markersize',15,'color',line_col);hold on;
    end
    ylim([-.75 1]);xlabel('Pulse Rate (pps)'); box off;xlim([0 350])
    if n_p == 1
        ylabel('Slope (sps/pps)');
    end
    if rec_mode 
     n_p = n_p+1;
    end
    set(gca,'fontsize',15)
end

end