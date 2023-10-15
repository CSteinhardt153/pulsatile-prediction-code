%pred_best_pr_for_target_fr.m
%For given current amplitude findings pr/fr mapping and predicts best pr
%sequence with minimal power consumption/pr vs. best match to fr sequence
%Started 10/14/21 by CRS
%Last Updated 10/15/21 by CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Finding best pr for a fixed I given a desired fr pattern:
%Choose 150 uA a common current amplitude - minimize on pr to make best fr
%(because for certain frs the same pr may max the best fr - minimal power
%consumption and correction for bends in relationship:
cd('..')
base_dir = pwd;
data_dir = fullfile(base_dir,'relevant_data')
%Produced Figure 5 A-D, Supplement Figure 4 ====================
%%PM around 200 pps
%Finding best pr to make similar fr: for the I,S combination
S = 30; % sps
I_cur = 250; %uA
pr_range = 0:.05:450;%0:.25:450;
[frs_change] = interp_pred_f_5_5_21(I_cur,S,pr_range);
mapped_frs = frs_change + S;

%HV_map
f_max = 350;
f_baseline = 100;
C = 5;% iregular 2 - regular?
HV_i = -450:450%[0:4095];%0-2048-4095 = [-450, 0, +450]
A = atanh(2*f_baseline./f_max -1);
use_PR = 0.5*f_max.*(1+tanh(A+C*((HV_i + 450)/450 - 1))); %firing rate for each head velocity

fr_t_HV = @(fr_i) 450*(1+ ((atanh((fr_i/(.5*f_max)) - 1) - A)/C)) - 450;

for n_HV =1:length(HV_i)
   [ a idx]= min(abs(use_PR(n_HV)-pr_range));
   fr_seq(n_HV) =mapped_frs(idx);
end
figure(1); subplot(1,2,1); plot(pr_range,mapped_frs,'k');box off;
ylabel('Firing Rate (sps)')
xlabel('Pulse Rate (pps)'); title(['I = ' num2str(I_cur)])
subplot(1,2,2);plot(HV_i,fr,'k'); hold on; 
plot(HV_i,fr_seq,'m'); box off;
plot(HV_i(1:max(find(fr < 110))),fr(1:max(find(fr < 110))),'b'); hold on;
plot([HV_i(max(find(fr < 110))) 500], [fr(max(find(fr < 110))) fr(max(find(fr < 110)))],'b')
xlabel('Head  Veloctiy (degree/s)'); ylabel('Firing Rate (sps)')
%title('Natural irregular firign rate to head velocity')

% plot(fr,HV_i,'r--')
% xlabel('Firing Rate (sps)'); ylabel('Head Velocity (degrees)')


%%
%Make moving firing rate trajectory:
dt = 1e-4;                   % seconds per sample
Fs = 1/dt; %s;%8000;                   % samples per second

StopTime = 1.5;             % seconds
t = (0:dt:StopTime-dt)';     % seconds
%%Sine wave:
Fc = 4;%4;                     % hertz
%fr_trajectory = 20*sin(2*pi*Fc*t)+40;%Fc = 4; 
fr_trajectory = 50*sin(2*pi*Fc*t)+55;%80*sin(2*pi*Fc*t)+120;
%fr_trajectory = 150 + 30*sin(2*pi*Fc*t) + 15*sin(2*pi*2.5*t + pi/5)+ 5*sin(2*pi*8*t + pi/3);
% Plot the signal versus time:
% figure(100);
% plot(t, fr_trajectory);
% xlabel('Time (in seconds)');
% ylabel('Firing Rate (sps)');
% title('Target Firing Rate versus Time');
fix_val = I_cur; fix_str = ' pps';
[best_minned_stim_pr best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,pr_range,t,fix_val,S,fix_str);

fix_str = 'uA';
%Use the solved from simulation to find best PAM:

best_or_no = 0; rate_mode = 1;

[out] = make_input_fr_pred(mu,S,I_cur,pr,t,best_minned_stim_pr,fr_trajectory,best_or_no,rate_mode)
%%
expt.num = [6];
run_mode = 'override';%  {'exact','override'}; %can run the exact experiment from study, override some parameters, or do a new experiment
%%% If choose override skip to line 109 to edit otherwise select experiment

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Deciding what outputs to visualize:
output.vis_plots = 1; %If want to see the afferent model APs without experiments
output.vis_plot_num = 6; %plot number when visualizing
output.label_aps = 0; %This is meaningless if don't set the seed, etc. - for check if AP spont or pulse induced
output.all_spk = 0;
output.do_phase_plane_anal = 0;
output.demo_pulse_mod = 0;
inj_cur = [1 1];

dt = 1e-6; %s
freq_plot = 1;

%Choose to do 150 +-50 for each and 120 +- 50 to show two diff extremes bot
%hexplainable w/ 1 second loops
firing = struct();
firing.sim_time = StopTime; %s

if (I_cur > 0)
    I_cur = I_cur/-20;
end
%Simulate it:
%mu=2 = 30, mu=5 = 12
[override] = set_overrides_v2(run_mode,output,...
    {'curr_options',I_cur},{'mu_IPT',2},{'inj_cur',inj_cur},{'is_reg',0},{'do_jitter',0},...
    {'tot_reps',1},{'sim_start_time',150},{'epsc_scale',1},{'sim_time',firing.sim_time*1e3});


firing.best_pr = best_minned_stim_pr;
firing.goal_fr = fr_trajectory;

firing.mod_f = firing.best_pr%fr_trajectory;%

firing.goal_fr_t = t;
%For sine: firing.pm_mod_amp*sin(firing.mod_freq*2*pi*t_full)+firing.pm_base;
override.sim_info.inj_cur = [1 0];
override.firing = firing;
override.firing.mod_timing = t;
override.firing.pm_base = [];
override.firing.pm_mod_amp = [];
override.firing.mod_freq = [];
override.sim_info.sim_time = override.firing.sim_time*1e3;
override.rate_mode = 1; % 1 = rate mode, 0 = amplitude mode
override.output.vis_plots = 0; %check pulses over time

%override.pulse_rate = [0:25:350];
override.tot_reps = 10;%10;
out = run_chosen_expt(expt,run_mode,override,output);
title(sprintf('Mapping for %s uA, S %s sps',num2str(-20*I_cur),num2str(S)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PAM around 300 pps
S = 11; % sps
I_range = 0:.5:350;%150; %uA
pr = 300;%% %0:.05:450;%0:.25:450
fix_str = ' pps';
frs_change = [];
for n_I = 1:length(I_range)
[frs_change(n_I)] = interp_pred_f_5_5_21(I_range(n_I),S,pr);
end
% Show mapping and find best input sequence for it:
mapped_frs = frs_change + S;
figure(90); plot(I_range,mapped_frs);
xlabel('I (uA)'); ylabel('Induced FR (sps)'); title('Mapping')
% Interp backwards to find best fit:
% unique_idxs = find(ismember(frs_change, unique(frs_change)));
% q = interp1(frs_change(unique_idxs),I_range(unique_idxs),5)

%%
%Make moving firing rate trajectory:
dt = 1e-4;                   % seconds per sample
Fs = 1/dt; %s;%8000;                   % samples per second

StopTime = 2;             % seconds
t = (0:dt:StopTime-dt)';     % seconds
%%Sine wave:
Fc = 3;                     % hertz
fr_trajectory = 50*sin(2*pi*Fc*t)+50;
fr_trajectory = 50 + 25*sin(2*pi*Fc*t) + 15*sin(2*pi*2.5*t + pi/5)+ 5*sin(2*pi*8*t + pi/3);
[best_minned_stim_I best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,I_range,t,pr,S,fix_str);

%% Speech data set example fr: 2/14/22
example_trace = [20 20 20 22 25 23 20 20 20 20 20 20 20 20 130 110 100 115 126 110 115 123 118 110  100 90 70 68 65 60 57 55 50 52 43 40 40 32 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30  30 30 30 30 30 30 30 30 30 30 30 30 30 40 50 55 60 80 73 70 68  75 68 65 62 60 58 57 55 57 58 60];
fin_trace = 1.4*(example_trace+ 7*rand(size(example_trace)));
%q = repmat(fin_trace,[40, 1]);
%fr_trajectory = 1.3*( q(:)+ 7*rand(size(q(:))));
full_len  = 3600;
t = 0:dt:(full_len-1)*dt;
fr_traj_full = interp1(linspace(t(1),t(end),length(fin_trace)),...
    fin_trace,linspace(t(1),t(end),full_len))
power_trace =    rand(1,38)+[0 0  0   5 21 3 21 24 10 5 3 2 0 3 0  0 0 0 0 0 0 0 0 0 0 0 0 3  2   0  3 1 5 2 1 0 1 0];
p_trace_full = interp1(linspace(t(1),t(end),length(power_trace)),...
    power_trace,linspace(t(1),t(end),length(t)));
figure(9); subplot(2,1,1);
plot(power_trace);
subplot(2,1,2); plot(p_trace_full);

figure(10); subplot(2,1,1);plot(example_trace+ 6*rand(size(example_trace)))
subplot(2,1,2);plot(fr_traj_full);

[best_minned_stim_I best_minned_stim_fr] = target_fr_to_best_pr(fr_traj_full, mapped_frs,I_range,t,pr,S,fix_str);

%Previously used for pr (pulse rate) only now using for pr or pa.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Deciding what outputs to visualize:
expt.num = [6];
run_mode = 'override';%  {'exact','override'}; %can run the exact experiment from study, override some parameters, or do a new experiment
%%% If choose override skip to line 109 to edit otherwise select experiment

output.vis_plots = 0; %If want to see the afferent model APs without experiments
output.vis_plot_num = 6; %plot number when visualizing
output.label_aps = 0; %This is meaningless if don't set the seed, etc. - for check if AP spont or pulse induced
output.all_spk = 0;
output.do_phase_plane_anal = 0;
output.demo_pulse_mod = 0;
inj_cur = [1 1];

dt = 1e-6; %s
freq_plot = 1;

%Choose to do 150 +-50 for each and 120 +- 50 to show two diff extremes bot
%hexplainable w/ 1 second loops
firing = struct();
firing.sim_time = StopTime; %s

if (I_cur > 0)
    I_cur = I_cur/-20;
end
%Simulate it:
%mu=2 = 30, mu=5 = 12
[override] = set_overrides_v2(run_mode,output,...
    {'curr_options',I_cur},{'mu_IPT',4},{'inj_cur',inj_cur},{'is_reg',0},{'do_jitter',0},...
    {'tot_reps',1},{'sim_start_time',150},{'epsc_scale',1},{'sim_time',firing.sim_time*1e3});

firing.best_I = best_minned_stim_I;
firing.goal_fr = fr_trajectory;

firing.mod_f = best_minned_stim_I;%
if sum(firing.mod_f == fr_trajectory) == length(firing.mod_f)
    case_str = 'Param = FR';
else
    case_str = 'Best Param';
end

firing.goal_fr_t = t;
%For sine: firing.pm_mod_amp*sin(firing.mod_freq*2*pi*t_full)+firing.pm_base;
%override.sim_info.inj_cur = [1 0];
override.firing = firing;
override.firing.mod_timing = t;
override.firing.pm_base = pr;
override.firing.pm_mod_amp = [];
override.firing.mod_freq = [];
override.firing.set_f=  1; % Set a function for current stimulation (rate_mode = 0)
override.sim_info.sim_time = override.firing.sim_time*1e3;
override.rate_mode = 0; % 1 = rate mode, 0 = amplitude mode
override.output.vis_plots = 0; %check pulses over time
override.tot_reps = 10;
out = run_chosen_expt(expt,run_mode,override,output);
title(sprintf('Mapping for %s uA, S %s sps, %s',num2str(-20*I_cur),num2str(S),case_str));

%%
% mus = [inf 4 2 1 0.5 0.25];
% Ss = [0 13 26 49 74 110];
% for n_mu = 1 :length(mus)
% expt.num =3; override.pulse_rate = 350;
% override.curr_options = -15:.1:0;
% 
% if isinf(mus(n_mu))
%     override.sim_info.inj_cur = [1 0];
% else
%      override.sim_info.inj_cur = [1 1]
%     override.sim_info.mu_IPT = mus(n_mu);
% end
% out = run_chosen_expt(expt,run_mode,override,output);
% frs(n_mu,:) = out.fr;
% end
% figure(10);plot(override.curr_options,frs)
% save('fr_pr_diff_I_for_pam_pred_detailed','frs','override','mus','Ss')
%% PAM Section
% % %q = load('fr_pr_diff_I_for_pam_pred_detailed.mat')
% % %%
% % figure (1);
% % col_map = winter(6);
% % for n_s = 1:6
% %     plot(override.curr_options*-20,frs(n_s,:),'color',col_map(n_s,:)); hold on;
% % end
% % xlabel('I'); ylabel('Firing Rate (sps)');
% % %4 - 13, 2 - 26, 1 - 49, 0.5 - 74, .25 - 110
data_dir_3_axon = fullfile(data_dir,'specific_runs/3_axon_runs_4_12_21');
cur_dir = data_dir_3_axon; %   

cd(cur_dir);
if strcmp(cur_dir,data_dir_3_axon)
    ax_fldrs = dir('axon_*')
    cd(ax_fldrs(1).name)
else
    ax_fldrs = 1;
end

diff_rate = dir('pr_fr_sim_I0-*')

s0 = dir('pr_fr_sim_I*_MInf*')
s0_info = load(s0.name);
col_map = winter(7);
pr = 350;
%Go through each current and spont rate:
for n_rats =  1:7
s_info = load(diff_rate(n_rats).name);
p_idx = find(s_info.pulse_rate == pr);
Ss(n_rats) = mean(s_info.fr(1,1,:));
mus(n_rats) = s_info.sim_info.mu_IPT;
frs(n_rats,:) = mean(s_info.fr(:,p_idx,:),3);
I_range = s_info.curr_options*-20;
figure(10);plot(-20*s_info.curr_options,mean(s_info.fr(:,p_idx,:),3),'-','color',col_map(n_rats,:)); hold on;
end
xlabel('I_stim'); ylabel('Firing Rate (sps)')
%
%override.pulse_rate = [0:25:350];
dt = 1e-4;                   % seconds per sample
Fs = 1/dt; %s;%8000;                   % samples per second
StopTime = 2;             % seconds
t = (0:dt:StopTime-dt)';     % seconds
%%Sine wave:
Fc = 4;%4;                     % hertz
%fr_trajectory = 20*sin(2*pi*Fc*t)+40;%Fc = 4; fr_trajectory = 80*sin(2*pi*Fc*t)+120;
fr_trajectory= 100 + 25*sin(2*pi*Fc*t) + 15*sin(2*pi*2.5*t + pi/5)+ 5*sin(2*pi*8*t + pi/3);
n_s =4;%5;
mapped_frs = frs(n_s,:);% I_range = override.curr_options*-20; S = Ss(n_s);
S = Ss(n_s); mu = mus(n_s);
fix_str = 'uA';
%Use the solved from simulation to find best PAM:

[best_minned_stim_I best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,I_range,t,pr,S,fix_str);
best_or_no = 1; rate_mode = 0;
I_cur =  nan; % because is PAM

%addpath(genpath('/Users/cynthiasteinhardt/Dropbox/code_directories/pulsatileDir'));
[out] = make_input_fr_pred(mu,S,I_cur,pr,t,best_minned_stim_I,fr_trajectory,best_or_no,rate_mode)


%% Compare with restoration of HV (head velocity)
f_max = 350;
f_baseline = 100;
C = 5;% iregular 2 - regular?
HV_i = -450:450;%[0:4095];%0-2048-4095 = [-450, 0, +450]
A = atanh(2*f_baseline./f_max -1);
fr = 0.5*f_max.*(1+tanh(A+C*((HV_i + 450)/450 - 1))); %firing rate for each head velocity

% figure(1); subplot(2,1,1); plot(HV_i,fr);
% xlabel('Head  Veloctiy (degrees)'); ylabel('Firing Rate (sps)')
% title('Natural irregular firign rate to head velocity')

fr_i = [0:350];
HV = 450*(1+ ((atanh((fr_i/(.5*f_max)) - 1) - A)/C)) - 450;
%  subplot(2,1,2); plot(fr_i,HV,'k'); hold on;
% plot(fr,HV_i,'r--')
% xlabel('Firing Rate (sps)'); ylabel('Head Velocity (degrees)')

fr_t_HV = @(fr_i) 450*(1+ ((atanh((fr_i/(.5*f_max)) - 1) - A)/C)) - 450;
%figure(20); plot(out.ts,out.pr_vect)

%%
col_map = winter(7);% same as in previous figures in rules of pulsatile stimulation
S_vals = [0 13.4 30.8 55.6 84.5 131.8];
fr_trajectory = [0 0.0001 1:.25:350];

figure(100);
alpha = 0.5;
for n_S = 1:length(S_vals)
%     subplot(length(S_vals),3,(n_S-1)*3 +1); plot(HV_i,fr,'k'); hold on;
%     subplot(length(S_vals),3,(n_S-1)*3 +2); plot(fr_trajectory, fr_trajectory,'k--'); hold on;
%     subplot(length(S_vals),3,(n_S-1)*3 +3); plot(fr_trajectory, fr_trajectory,'k--'); hold on;
%     
   subplot(1,3,1); plot(HV_i,fr,'k'); hold on;
   subplot(1,3,2); plot(fr_trajectory, fr_trajectory,'k--'); hold on;
   subplot(1,3,3);  plot(fr_trajectory, fr_trajectory,'k--'); hold on;
    
    S = S_vals(n_S);%0; % sps
    I_cur = 150; %uA
    pr_range = 0:.01:450;%0:.25:450
    
    [frs_change] = interp_pred_f_5_5_21(I_cur,S,pr_range);
    mapped_frs = frs_change + S;
     figure(90); plot(pr_range,mapped_frs,'color',col_map(n_S,:)); hold on;
    xlabel('PR (pps)'); ylabel('Induced FR (sps)'); title('Mapping')

  
    
    %Make moving firing rate trajectory:                    % hertz
    t=1:length(fr_trajectory); 
    [best_minned_stim_pr best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,pr_range,t,I_cur,S, fix_str);
    
    HV_pred = fr_t_HV(best_minned_stim_fr);
    %subplot(length(S_vals),3,(n_S-1)*3 + 1); 
    subplot(1,3,1); 
    plot(max(-500,fr_t_HV([min(best_minned_stim_fr) max(best_minned_stim_fr)])),...
        [min(best_minned_stim_fr) max(best_minned_stim_fr)],'*','color',col_map(n_S,:));%'color',[col_map(n_S,:) alpha],'linewidth',2);
    xlabel('Head  Veloctiy (degrees)'); ylabel('Firing Rate (sps)')
    title('Natural irregular firign rate to head velocity')
     box off;
    subplot(1,3,3);
    plot(fr_trajectory,best_minned_stim_pr,'color',[col_map(n_S,:) alpha],'linewidth',2); hold on;
    xlabel('Target Firing Rate (sps)'); ylabel('Best Pulse Rate (pps)');
    box off;
    xlim([0 350])
    subplot(1,3,2);
    plot(fr_trajectory,best_minned_stim_fr,'color',[col_map(n_S,:) alpha],'linewidth',2); hold on;
    
%     plot(fr_trajectory,best_minned_stim_fr,'color',[col_map(n_S,:) alpha],'linewidth',2); hold on;
    xlabel('Best Pulse Rate (pps)');ylabel('Best Firing Rate Reconstruction (sps)')
    ylim([0 350]);
     box off;
end

%% PAM
col_map = flipud(winter(6));% same as in previous figures 
S_vals = [0 13.4 30.8 55.6 84.5 131.8];
fr_trajectory = [0 0.0001 1:.25:350];

alpha = 0.5;
I_range = 0.1:.1:350;%150; %uA
pr = 250;%% %0:.05:450;%0:.25:450

for n_S = 1:length(S_vals)

    S = S_vals(n_S);%0; % sps
        
    fix_str = ' pps';
    frs_change = [];
    parfor n_I = 1:length(I_range)
        [frs_change(n_I)] = interp_pred_f_5_5_21(I_range(n_I),S,pr);
    end
    
    mapped_frs = frs_change + S;
%     figure(90); plot(I_range,mapped_frs,'color',col_map(n_S,:)); hold on;
%     xlabel('I (uA)'); ylabel('Induced FR (sps)'); title('Mapping')

    %Make moving firing rate trajectory:                    % hertz
    figure(101);
%     subplot(length(S_vals),3,(n_S-1)*3 +1); plot(HV_i,fr,'k'); hold on;
%     subplot(length(S_vals),3,(n_S-1)*3 +2); plot(fr_trajectory, fr_trajectory,'k--'); hold on;
%     subplot(length(S_vals),3,(n_S-1)*3 +3); plot(fr_trajectory, fr_trajectory,'k--'); hold on;
%     
    subplot(1,3,1); plot(HV_i,fr,'k'); hold on;
    subplot(1,3,2); plot(fr_trajectory, fr_trajectory,'k--'); hold on;
    subplot(1,3,3); plot(fr_trajectory, fr_trajectory,'k--'); hold on;
  
    t=1:length(fr_trajectory); 
    [best_minned_stim_param best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,I_range,t,pr,S, fix_str);
    
    HV_pred = fr_t_HV(best_minned_stim_fr);
    subplot(length(S_vals),3,(n_S-1)*3 + 1); 
    subplot(1,3,1); 
     plot(max(-500,fr_t_HV([min(best_minned_stim_fr) max(best_minned_stim_fr)])),...
        [min(best_minned_stim_fr) max(best_minned_stim_fr)],'*','color',col_map(n_S,:));%'color',[col_map(n_S,:) alpha],'linewidth',2);

    %plot(HV_pred,best_minned_stim_fr,'color',[col_map(n_S,:) alpha],'linewidth',2);
    xlabel('Head  Veloctiy (degrees)'); ylabel('Firing Rate (sps)')
    title('Natural irregular firign rate to head velocity')
     box off;
    subplot(1,3,3);
    %subplot(length(S_vals),3,(n_S-1)*3 + 3); 
    plot(fr_trajectory,best_minned_stim_param,'color',[col_map(n_S,:) alpha],'linewidth',2); hold on;
    xlabel('Target Firing Rate (sps)'); ylabel('Best Pulse Rate (pps)');
    box off;
    xlim([0 350])
    subplot(1,3,2);
    %subplot(length(S_vals),3,(n_S-1)*3 + 2); 
    plot(fr_trajectory,best_minned_stim_fr,'color',[col_map(n_S,:) alpha],'linewidth',2); hold on;
    xlabel('Target Firing Rate (pps)');ylabel('Best Firing Rate Reconstruction (sps)')
    ylim([0 350]);
     box off;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Functions for above
function [best_minned_stim_in best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,input_range,t,fix_val,S,fix_str)
%Previously used for pr (pulse rate) only now using for pr or pa.
%Instead of pr input called input. Output still called "in" just to be clear
%vs. fr
plot_it = 1; plot_all_steps =0;
refresh_interv = 1;%(Fs/pr_fs);
n_cnt= 1;
for n_targ_ts = 1:refresh_interv:length(fr_trajectory)
    
    less_in_idx = find(mapped_frs <= fr_trajectory(n_targ_ts));
    [less_mses less_idx]=sort(fr_trajectory(n_targ_ts) - mapped_frs(less_in_idx));
    greater_in_idx = find(mapped_frs >= fr_trajectory(n_targ_ts));
    [greater_mses greater_idx]=sort(abs(fr_trajectory(n_targ_ts) - mapped_frs(greater_in_idx)));
    
    if (fr_trajectory(n_targ_ts) <= min(mapped_frs))
        if isempty(less_idx)
            [a idx] = min(mapped_frs);
            poten_ins = input_range(idx);
            idx_best = idx;
        else
            poten_ins = min(input_range(less_idx));
            idx_best = min(less_idx);
        end
        best_minned_stim_in(n_cnt) = poten_ins;
        best_minned_stim_fr(n_cnt) = mapped_frs(idx_best);
    else
        
        rel_greater= greater_in_idx(greater_idx);
        try_n = 6;
        rel_less= less_in_idx(less_idx(1:min(length(less_idx),try_n)));
        rel_great= greater_in_idx(greater_idx(1:min(length(greater_idx),try_n)));
        if isempty(rel_great)
            poten_ins = input_range(less_idx(1));
            best_minned_stim_in(n_cnt) = poten_ins;
            best_minned_stim_fr(n_cnt) = mapped_frs(less_idx(1));
        else
            n_pairs = 1;
            clear pairs_in
            for n_t = 1:min(length(less_idx),try_n)
                [match_greater]=find(abs(rel_less(n_t) - rel_great) <=1);
                if isempty(match_greater)
                else
                    if length(match_greater) > 1
                        match_greater = min(match_greater(rel_great(match_greater) ~= rel_less(n_t)));
                        pair_ins(n_pairs,:) = [rel_less(n_t) rel_great(match_greater)];
                        n_pairs = n_pairs + 1;
                        
                    else
                        pair_ins(n_pairs,:) = [rel_less(n_t) rel_great(match_greater)];
                        n_pairs = n_pairs + 1;
                    end
                end
            end
            
            %plot(input_range(greater_in_idx(greater_idx)),mapped_frs(greater_in_idx(greater_idx)),'ro');
            %plot(input_range(less_in_idx(less_idx)),mapped_frs(less_in_idx(less_idx)),'bo');
            poten_ins = [];
            for n_p  =1:(n_pairs-1)
                yx_rev = polyfit(input_range(pair_ins(n_p,:)),mapped_frs(pair_ins(n_p,:)),1);
                rev_vars = [1/yx_rev(1) -yx_rev(2)/yx_rev(1)];
                in_var = polyval(rev_vars,fr_trajectory(n_targ_ts));
                poten_ins(n_p) = in_var;
                if plot_all_steps
                    figure(7); plot(input_range,mapped_frs,'.-'); hold on;
                    plot([0 400],[fr_trajectory(n_targ_ts) fr_trajectory(n_targ_ts)],'k');
                    %plot(input_range(pair_ins(n_p,1)),mapped_frs(pair_ins(n_p,1)),'*');
                    %plot(input_range(pair_ins(n_p,2)),mapped_frs(pair_ins(n_p,2)),'*');
                    plot(in_var,fr_trajectory(n_targ_ts),'x');
                end
            end
            if isempty(poten_ins)
                poten_ins = nan;
            end
            best_minned_stim_in(n_cnt) = (min(poten_ins));
            best_minned_stim_fr(n_cnt) = fr_trajectory(n_targ_ts);
            if plot_all_steps
               plot(best_minned_stim_in(n_cnt),best_minned_stim_fr(n_cnt),'ro') 
            end
            
        end
    end
   
                           
    
% %     [diff min_idx]=min(abs(fr_trajectory(n_targ_ts) - mapped_frs));
% %     [diffs sort_idxs]=sort(abs(fr_trajectory(n_targ_ts) - mapped_frs),'ascend');
% %     best_stim_in(n_cnt) = input_range(min_idx);
% %     best_stim_fr(n_cnt) = mapped_frs(min_idx);

% %     %Add in min on pr:
% %     in_min_err = ((input_range(sort_idxs)/400)*[max(diffs)]*.08  + diffs);
% %     [min_weighted_err best_minned_in_idx]= min(in_min_err);
% %     best_minned_stim_in(n_cnt) = input_range(sort_idxs(best_minned_in_idx));
% %     best_minned_stim_fr(n_cnt) = mapped_frs(sort_idxs(best_minned_in_idx));
    % figure(6); plot(diffs); hold on;
    % plot((pr_range/400)*[max(diffs)]*.25  + diffs);
    t_stims(n_cnt) =t(n_targ_ts);
    n_cnt = n_cnt+ 1;
end
%figure(105);
if plot_it
    figure(103); %subplot(3,1,1);
    subplot(2,1,1);
    plot(input_range,mapped_frs);
    title(sprintf('Mapping for %s %s, S %s sps',num2str(fix_val),fix_str, num2str(S)));
    xlabel('PR (pps)'); ylabel('FR (sps)'); box off;
%     subplot(3,2,3); plot(t_stims,best_stim_in,'b'); hold on; box off;
%     % xlim([550 1300]*1e-3)
%     title('Absolute minimum error in fr')
    %subplot(3,2,4);
%     
%     plot(t, fr_trajectory,'k'); hold on;
%     plot(t_stims,best_stim_fr,'b'); hold on; box off;
    % xlim([550 1300]*1e-3)
    %figure(106);
    %subplot(3,2,5);
    subplot(2,2,3);
    plot(t_stims,best_minned_stim_in,'r');box off;
    title(['Minimizing ' fix_str ])
    % xlim([550 1300]*1e-3)
    %subplot(3,2,6);
     subplot(2,2,4);
    plot(t, fr_trajectory,'k'); hold on;
    plot(t_stims,best_minned_stim_fr,'b'); hold on;
    % xlim([550 1300]*1e-3)
end

do_vid =1;
if do_vid
newVid = VideoWriter('speech_conversion_2_14_22_fr_pulse', 'MPEG-4');
%VideoWriter('vid_corr_pr_fr_11_11_21', 'MPEG-4'); % New
fps= 20;
newVid.FrameRate = fps;
newVid.Quality = 100;
open(newVid);

pulse_time_sampling = round((1/350)/t(2));% not going ot use right now

vid_step =  30;
t_rel = 1:vid_step:length(best_minned_stim_in);
    figure(900);clf;
    set(gcf,'color','w');
for n_vid = 1:vid_step:length(best_minned_stim_in)
%     figure(900);
%     subplot(3,1,1); cla;
%     plot(input_range,mapped_frs,'k-'); hold on;
%     plot(best_minned_stim_in(n_vid),best_minned_stim_fr(n_vid),'r.','markersize',10);
%     subplot(3,1,2);%plot(t(t_rel),fr_trajectory(t_rel),'k'); hold on;
%     plot(t(1:vid_step:n_vid),best_minned_stim_fr(1:vid_step:n_vid),'k','linewidth',2) 
%      xlim([0 0.4]);
%     subplot(3,1,3);
%     plot([t(n_vid) t(n_vid)],[-(best_minned_stim_in(n_vid)/2) (best_minned_stim_in(n_vid)/2)],'k'); hold on;
%     xlim([0 0.4]);
%     %for loop overpulses
%     %plot(t(t_rel),best_minned_stim_in(t_rel),'k'); hold on;
%     %plot(t(1:vid_step:n_vid),best_minned_stim_in(1:vid_step:n_vid),'b','linewidth',2);

    figure(900);
    subplot(1,2,1); set(gca,'fontsize',18); 
    plot(t(1:vid_step:n_vid),best_minned_stim_fr(1:vid_step:n_vid),'k','linewidth',2) 
    xlim([0 0.4]); box off; ylabel('Firing Rate (sps)')
     xlabel('Time (ms)');
     %ylabel('Power (W/Hz)'); ylim([0 25]);
     ylim([0 200]);
     set(gca,'fontsize',18)
    subplot(1,2,2);
     ylim([-140 140])
     plot([0 0.4],[0 0],'k'); hold on;
    plot([t(n_vid) t(n_vid)],[-(best_minned_stim_in(n_vid)) (best_minned_stim_in(n_vid))],'k'); hold on;
   
    xlim([0 0.4]); box off; ylabel('Pulse Amplitude (uA)');xlabel('Time (ms)')
   set(gca,'fontsize',18);
    %for loop overpulses
    %plot(t(t_rel),best_minned_stim_in(t_rel),'k'); hold on;
    %plot(t(1:vid_step:n_vid),best_minned_stim_in(1:vid_step:n_vid),'b','linewidth',2);

    frame = getframe(gcf);
    writeVideo(newVid,frame);
   
   %turn into pulse train ever
end
close(newVid)
end

end
%%
%where plot the moving averaged firing rate produced by the pulsatile
%stimulation paradigm
function [out] = make_input_fr_pred(mu,S,I_cur,pr,t,best_minned_stim_I,fr_trajectory,best_or_no,rate_mode)
%%%%best_or_no = 1 - best pred, no fr= param
%Deciding what outputs to visualize:
expt.num = [6];
run_mode = 'override';%  {'exact','override'}; %can run the exact experiment from study, override some parameters, or do a new experiment
%%% If choose override skip to line 109 to edit otherwise select experiment

output.vis_plots = 0; %If want to see the afferent model APs without experiments
output.vis_plot_num = 6; %plot number when visualizing
output.label_aps = 0; %This is meaningless if don't set the seed, etc. - for check if AP spont or pulse induced
output.all_spk = 0;
output.do_phase_plane_anal = 0;
output.demo_pulse_mod = 0;
inj_cur = [1 1];

dt = 1e-6; %s

%Choose to do 150 +-50 for each and 120 +- 50 to show two diff extremes bot
%hexplainable w/ 1 second loops
firing = struct();
firing.sim_time = t(end); %s

if (I_cur > 0)
    I_cur = I_cur/-20;
end
%Simulate it:
%mu=2 = 30, mu=5 = 12
[override] = set_overrides_v2(run_mode,output,...
    {'curr_options',I_cur},{'mu_IPT',mu},{'inj_cur',inj_cur},{'is_reg',0},{'do_jitter',0},...
    {'tot_reps',1},{'sim_start_time',150},{'epsc_scale',1},{'sim_time',firing.sim_time*1e3});

override.rate_mode = rate_mode; 

firing.goal_fr = fr_trajectory;
if best_or_no
    if rate_mode
        firing.mod_f = best_minned_stim_I;%
        override.firing.best_pr = firing.mod_f;
    else
        firing.mod_f = best_minned_stim_I;%
        override.firing.best_I = firing.mod_f;
    end
else
    firing.mod_f = fr_trajectory;%
end
if sum(firing.mod_f == fr_trajectory) == length(firing.mod_f)
    case_str = 'Param = FR';
else
    case_str = 'Best Param';
end

firing.goal_fr_t = t;
%For sine: firing.pm_mod_amp*sin(firing.mod_freq*2*pi*t_full)+firing.pm_base;
%override.sim_info.inj_cur = [1 0];
override.firing = firing;
override.firing.mod_timing = t;
override.firing.pm_base = pr;
override.firing.pm_mod_amp = [];
override.firing.mod_freq = [];
override.firing.set_f=  1; % Set a function for current stimulation (rate_mode = 0)
override.sim_info.sim_time = override.firing.sim_time*1e3;
override.tot_reps = 10;
out = run_chosen_expt(expt,run_mode,override,output);
title(sprintf('Mapping for %s uA, S %s sps, %s',num2str(-20*I_cur),num2str(S),case_str));
end