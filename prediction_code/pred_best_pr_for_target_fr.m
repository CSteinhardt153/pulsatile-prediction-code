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

%addpath(genpath('/Users/cynthiasteinhardt/Dropbox/code_directories/pulsatileDir/simulation_code/'));

%%PM around 200 pps
%Finding best pr to make similar fr: for the I,S combination
S = 30; % sps
I_cur = 150; %uA
pr_range = 0:.05:450;%0:.25:450;
[frs_change] = interp_pred_f_5_5_21(I_cur,S,pr_range);
mapped_frs = frs_change + S;


%%

%Make moving firing rate trajectory:
dt = 1e-4;                   % seconds per sample
Fs = 1/dt; %s;%8000;                   % samples per second

StopTime = 2;             % seconds
t = (0:dt:StopTime-dt)';     % seconds
%%Sine wave:
Fc = 4;                     % hertz
fr_trajectory = 80*sin(2*pi*Fc*t)+120;
%fr_trajectory = 150 + 25*sin(2*pi*Fc*t) + 15*sin(2*pi*2.5*t + pi/5)+ 5*sin(2*pi*8*t + pi/3);
% Plot the signal versus time:
% figure(100);
% plot(t, fr_trajectory);
% xlabel('Time (in seconds)');
% ylabel('Firing Rate (sps)');
% title('Target Firing Rate versus Time');
fix_val = I_cur; fix_str = ' uA';
[best_minned_stim_pr best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,pr_range,t,fix_val,S,fix_str)

%%
expt.num = [6];
run_mode = 'override';%  {'exact','override'}; %can run the exact experiment from study, override some parameters, or do a new experiment
%%% If choose override skip to line 109 to edit otherwise select experiment

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Deciding what outputs to visualize:
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
    {'curr_options',I_cur},{'mu_IPT',2},{'inj_cur',inj_cur},{'is_reg',0},{'do_jitter',0},...
    {'tot_reps',1},{'sim_start_time',150},{'epsc_scale',1},{'sim_time',firing.sim_time*1e3});


firing.best_pr = best_minned_stim_pr;
firing.goal_fr = fr_trajectory;

firing.mod_f = firing.best_pr%fr_trajectory;%

firing.goal_fr_t = t;
%For sine: firing.pm_mod_amp*sin(firing.mod_freq*2*pi*t_full)+firing.pm_base;
%override.sim_info.inj_cur = [1 0];
override.firing = firing;
override.firing.mod_timing = t;
override.firing.pm_base = [];
override.firing.pm_mod_amp = [];
override.firing.mod_freq = [];
override.sim_info.sim_time = override.firing.sim_time*1e3;
override.rate_mode = 1; % 1 = rate mode, 0 = amplitude mode
override.output.vis_plots = 0; %check pulses over time

%override.pulse_rate = [0:25:350];
override.tot_reps = 10;
out = run_chosen_expt(expt,run_mode,override,output);
title(sprintf('Mapping for %s uA, S %s sps',num2str(-20*I_cur),num2str(S)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PAM around 200 pps
S = 30; % sps
I_range = 0:.5:350;%150; %uA
pr = 100;%% %0:.05:450;%0:.25:450
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
%fr_trajectory = 50 + 25*sin(2*pi*Fc*t) + 15*sin(2*pi*2.5*t + pi/5)+ 5*sin(2*pi*8*t + pi/3);

[best_minned_stim_I best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,I_range,t,pr,S,fix_str);
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
    {'curr_options',I_cur},{'mu_IPT',2},{'inj_cur',inj_cur},{'is_reg',0},{'do_jitter',0},...
    {'tot_reps',1},{'sim_start_time',150},{'epsc_scale',1},{'sim_time',firing.sim_time*1e3});


firing.best_I = best_minned_stim_I;
firing.goal_fr = fr_trajectory;

firing.mod_f = fr_trajectory;%best_minned_stim_I;%
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
override.output.vis_plots = 1; %check pulses over time

%override.pulse_rate = [0:25:350];
override.tot_reps = 1;%10;
out = run_chosen_expt(expt,run_mode,override,output);
title(sprintf('Mapping for %s uA, S %s sps, %s',num2str(-20*I_cur),num2str(S),case_str));
%% Compare with restoration of HV (head velocity)
f_max = 350;
f_baseline = 100;
C = 5;% iregular 2 - regular?
HV_i = -450:450%[0:4095];%0-2048-4095 = [-450, 0, +450]
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
    subplot(1,3,3); plot(fr_trajectory, fr_trajectory,'k--'); hold on;
    
    S = S_vals(n_S);%0; % sps
    I_cur = 150; %uA
    pr_range = 0:.01:450;%0:.25:450
    
    [frs_change] = interp_pred_f_5_5_21(I_cur,S,pr_range);
    mapped_frs = frs_change + S;
    
    
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
col_map = winter(7);% same as in previous figures in rules of pulsatile stimulation
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
%     figure(90); plot(I_range,mapped_frs);
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
    [best_minned_stim_pr best_minned_stim_fr] = target_fr_to_best_pr(fr_trajectory, mapped_frs,I_range,t,pr,S, fix_str);
    
    HV_pred = fr_t_HV(best_minned_stim_fr);
    %subplot(length(S_vals),3,(n_S-1)*3 + 1); 
    subplot(1,3,1); 
     plot(max(-500,fr_t_HV([min(best_minned_stim_fr) max(best_minned_stim_fr)])),...
        [min(best_minned_stim_fr) max(best_minned_stim_fr)],'*','color',col_map(n_S,:));%'color',[col_map(n_S,:) alpha],'linewidth',2);

    %plot(HV_pred,best_minned_stim_fr,'color',[col_map(n_S,:) alpha],'linewidth',2);
    xlabel('Head  Veloctiy (degrees)'); ylabel('Firing Rate (sps)')
    title('Natural irregular firign rate to head velocity')
     box off;
    subplot(1,3,3);
    %subplot(length(S_vals),3,(n_S-1)*3 + 3); 
    plot(fr_trajectory,best_minned_stim_pr,'color',[col_map(n_S,:) alpha],'linewidth',2); hold on;
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
plot_it = 0;
refresh_interv = 1;%(Fs/pr_fs);
n_cnt= 1;
for n_targ_ts = 1:refresh_interv:length(fr_trajectory)
    [diff min_idx]=min(abs(fr_trajectory(n_targ_ts) - mapped_frs));
    [diffs sort_idxs]=sort(abs(fr_trajectory(n_targ_ts) - mapped_frs),'ascend');
    best_stim_in(n_cnt) = input_range(min_idx);
    best_stim_fr(n_cnt) = mapped_frs(min_idx);
    t_stims(n_cnt) =t(n_targ_ts);
    %Add in min on pr:
    in_min_err = ((input_range(sort_idxs)/400)*[max(diffs)]*.08  + diffs);
    [min_weighted_err best_minned_in_idx]= min(in_min_err);
    best_minned_stim_in(n_cnt) = input_range(sort_idxs(best_minned_in_idx));
    best_minned_stim_fr(n_cnt) = mapped_frs(sort_idxs(best_minned_in_idx));
    % figure(6); plot(diffs); hold on;
    % plot((pr_range/400)*[max(diffs)]*.25  + diffs);
    n_cnt = n_cnt+ 1;
end
%figure(105);
if plot_it
    figure(103); subplot(3,1,1); plot(input_range,mapped_frs);
    title(sprintf('Mapping for %s %s, S %s sps',num2str(fix_val),fix_str, num2str(S)));
    xlabel('PR (pps)'); ylabel('FR (sps)'); box off;
    subplot(3,2,3); plot(t_stims,best_stim_in,'b'); hold on; box off;
    % xlim([550 1300]*1e-3)
    title('Absolute minimum error in fr')
    subplot(3,2,4); plot(t, fr_trajectory,'k'); hold on;
    plot(t_stims,best_stim_fr,'b'); hold on; box off;
    % xlim([550 1300]*1e-3)
    %figure(106);
    subplot(3,2,5);
    plot(t_stims,best_minned_stim_in,'r');box off;
    title(['Minimizing ' fix_str ])
    % xlim([550 1300]*1e-3)
    subplot(3,2,6); plot(t, fr_trajectory,'k'); hold on;
    plot(t_stims,best_minned_stim_fr,'b'); hold on;
    % xlim([550 1300]*1e-3)
end
end