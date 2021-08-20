%pp_sp_sort.m
%Program to sort spikes within a pulsatile stimulation experiments by
%categorizign spontanoues spikes and spikes within the artifact time of
%pulses
%8/20 update - add check of size of spont spike on averaged normalized
%scale, put on git
%%Started 7/26/21
%Last Updated 8/20/21
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

plot_outputs = 1; % Plot the final fittings/ spike sorting
plot_steps = 1; % Plot the intermediates (kaiser threshold, etc. good for fitting new data)


data_dir = '/Users/cynthiasteinhardt/Dropbox/Fridman_lab/submissions/pulsatile/diana_raw_data_reanalysis_7_26_21';
cd(data_dir);
file_names = dir('rawdata_aff*.mat');

for n_dat_fs = 1:length(file_names)
disp(sprintf('Starting new afferent: %s', file_names(n_dat_fs).name ))
aff_dat = load(file_names(n_dat_fs).name);
stim_trace = aff_dat.stim;
raw_spiking = aff_dat.rawspk;

if  strcmp(file_names(n_dat_fs).name, 'rawdata_afferent7.mat')
    raw_spiking= -aff_dat.rawspk;
end

%Plot the raw:
plot_it = 1;
if plot_it
    figure;
    ax(1) = subplot(2,1,1);
    plot(stim_trace,'r');
        title(file_names(n_dat_fs).name(9:end-4)); ylabel('V (mV)')
     ax(2) = subplot(2,1,2);
     plot(raw_spiking,'k');hold on;
     linkaxes(ax,'x');

end

end

%% Pulse-induced spike detection
% Artifact extraction strategy
% Alone to timing of spike delivery
%Choose window around spike time
% PCA this window
%Group into Spike/ no spike

%% Settings for finding pulse times and therefore spontanoues AP times
% Look at window outside pulse time
% Set threshold min and max - find the Kaiser filter timing that shows
% spike
do_aff_1= 1;
if do_aff_1
    cur_timing = aff_1.t;
    cur_spiking = aff_1.rawspk;
    cur_stim =aff_1.stim;
    
    k_thresh_min_max = [2.5e5 4e5];
    % pulse_art_wind = [50];
    pulse_art_wind = 80;
    aff_num = 1;
else
    cur_timing = aff_7.t;%aff_1.t;
    cur_spiking = -aff_7.rawspk; %aff_1.rawspk;
    cur_stim = aff_7.stim;%aff_1.stim;
    k_thresh_min_max = [1e3 2e3];
    pulse_art_wind = [120];
    aff_num = 7;
end

plot_it = 1;
spike_info = find_AP_times(cur_timing,cur_spiking,cur_stim, k_thresh_min_max,pulse_art_wind,32,plot_it)
%subplot(3,1,1); title(['Afferent ' num2str(aff_num)])
figure(10);
plot(cur_timing, cur_spiking,'k'); hold on;
plot(cur_timing(spike_info.pulse_evoked_spk_idxs),cur_spiking(spike_info.pulse_evoked_spk_idxs),'m.');
plot(cur_timing(spike_info.spont_spk_idxs),cur_spiking(spike_info.spont_spk_idxs),'g.');

%% Check induced firing rate:
new_block_cutoff = 1e4;
pulse_blck_breaks = find(diff(spike_info.pulses_tot) > new_block_cutoff);

clear pulse_regions
pulse_regions(1,:) = [spike_info.pulses_tot(1) spike_info.pulses_tot(pulse_blck_breaks(1))];
for n_p_reg= 1:length(pulse_blck_breaks)-1
    [pulse_blck_breaks(n_p_reg)+1 pulse_blck_breaks(n_p_reg+1)]
    pulse_regions(n_p_reg+1,:) = [spike_info.pulses_tot(pulse_blck_breaks(n_p_reg)+1) spike_info.pulses_tot(pulse_blck_breaks(n_p_reg+1))];
end
pulse_regions(n_p_reg+2,:) = [spike_info.pulses_tot(pulse_blck_breaks(end)+1) spike_info.pulses_tot(end)];

%Correct pulse blocks:
figure(11);
plot(cur_timing, cur_spiking,'k'); hold on;

if do_aff_1
    time_diffs=( (cur_timing(pulse_regions(:,2)) - cur_timing(pulse_regions(:,1)))); %ms --> s
else
    time_diffs=( (cur_timing(pulse_regions(:,2)) - cur_timing(pulse_regions(:,1))))*1e-3; %ms --> s
end
use_winds = (time_diffs ~= 0);

for n_p_reg= 1:size(pulse_regions,1)
    if ismember(n_p_reg,find(use_winds))
        plot(cur_timing(pulse_regions(n_p_reg,1)),ones(size(pulse_blck_breaks)),'ro')
        plot(cur_timing(pulse_regions(n_p_reg,2)),ones(size(pulse_blck_breaks)),'r*')
    end
end

%%
%Num pulses per block:
clear n_pulses pulsespk spontspk spont_winds
for n_p_blcks = 1:size(pulse_regions,1)
    n_pulses(n_p_blcks) = sum((spike_info.pulses_tot < pulse_regions(n_p_blcks,2)) & ...
        (spike_info.pulses_tot > pulse_regions(n_p_blcks,1)));
    pulsespk(n_p_blcks) = sum((spike_info.pulse_evoked_spk_idxs < pulse_regions(n_p_blcks,2)) & ...
        (spike_info.pulse_evoked_spk_idxs > pulse_regions(n_p_blcks,1)));
    spontspk(n_p_blcks) = sum((spike_info.spont_spk_idxs < pulse_regions(n_p_blcks,2)) & ...
        (spike_info.spont_spk_idxs > pulse_regions(n_p_blcks,1)));
    if (n_p_blcks < size(pulse_regions,1))
        spont_winds(n_p_blcks) = sum((spike_info.spont_spk_idxs > pulse_regions(n_p_blcks,2)) & ...
            (spike_info.spont_spk_idxs < pulse_regions(n_p_blcks+1,1)));
    end
end
spont_winds(n_p_blcks)= sum(spike_info.spont_spk_idxs > pulse_regions(n_p_blcks,2));


%%
n_ps = n_pulses(use_winds);
time_ds = time_diffs(use_winds);
p_spks = pulsespk(use_winds);
spont_spks = spontspk(use_winds);
spont_win = spont_winds(use_winds);
figure(41); subplot(2,1,1);
plot(n_ps./time_ds,(p_spks + spont_spks)./time_ds,'k.'); hold on;
plot(zeros(size(spont_win)),spont_win./time_ds,'r.');
ylabel('Firng Rate (sps)'); xlabel('Pulse Rate (pps)')
set(gca,'fontsize',16)
subplot(2,1,2);


%Group by pulse rate:
pulse_rates = round(n_ps./time_ds);
un_prs = unique(pulse_rates);
for n_un_prs  = 1:length(unique(pulse_rates))
    pr_info(n_un_prs).idx = find(abs(pulse_rates - un_prs(n_un_prs)) < 1);
    pr_info(n_un_prs).pr_val = un_prs(n_un_prs);
    
    pr_idxs =  pr_info(n_un_prs).idx;
    errorbar(0,mean(spont_win./time_ds),...
        std(spont_win./time_ds),'r.-'); hold on;
    errorbar(mean(n_ps(pr_idxs)./time_ds(pr_idxs)),...
        mean((p_spks( pr_idxs) + spont_spks( pr_idxs))./time_ds( pr_idxs)),...
        std((p_spks( pr_idxs) + spont_spks( pr_idxs))./time_ds( pr_idxs)),'k.');
    avg_frs(n_un_prs) =mean((p_spks( pr_idxs) + spont_spks( pr_idxs))./time_ds( pr_idxs));
end

% Predict with model:
addpath('/Users/cynthiasteinhardt/Dropbox/single-neuron-stim-model/vestibular-neuron-models/vest_model_pulsatile/simpler_format')
[I_idx,rms_best,fr_pred_best] = two_d_rms_eval(mean(spont_win./time_ds), ([pr_info.pr_val]'),avg_frs);

plot(0:400,fr_pred_best,'b'); hold on;
[short_pred] = interp_pred_f_5_5_21(I_idx,mean(spont_win./time_ds),([pr_info.pr_val]));
[full_pred] = interp_pred_f_5_5_21(I_idx,mean(spont_win./time_ds),[0:350]);

%plot([0 [pr_info.pr_val]],mean(spont_win./time_ds) + [0 short_pred],'o','color','b');%col_dir(amp_grp+col_shift,:,:));
plot([0:350],mean(spont_win./time_ds) +full_pred,':','color','m');%col_dir(amp_grp+col_shift,:,:),'linewidth',1);
set(gca,'fontsize',16)

function [spike_info] = find_AP_times(cur_timing,cur_spiking,cur_stim, k_thresh_min_max,pulse_art_wind,fig_num,plot_it)
%Find Pulse times:
thresh_pulse = 1000;% voltes
min_inter_event_steps = 20;
min_inter_event_time = 20*(cur_timing(2) - cur_timing(1));

%Find time of start pulses for alignment and also for not counting pulses
%in spontaneous firing
[pulse_start_idxs] = detect_same_event(cur_stim,cur_stim,cur_timing, thresh_pulse,min_inter_event_steps);
% Find the same point (peak within the pulse start window):
% plot(cur_timing(pulse_start_idxs(1:10)),cur_stim(pulse_start_idxs(1:10)),'go')

surround_wind = [20 80]; % idxs
clear pulse_start
for n_p_idx = 1:length(pulse_start_idxs)
    search_wind = (pulse_start_idxs(n_p_idx) - surround_wind(1)):(pulse_start_idxs(n_p_idx) + surround_wind(2));
    stim_reference = cur_stim(search_wind);
    
    for n_step =2:(length(stim_reference)-1)
        kaiser_fact_stim(n_step) = stim_reference(n_step)^2 - stim_reference(n_step - 1) - stim_reference(n_step +1);
    end
    start_pulse_idx = min(find(diff(diff(kaiser_fact_stim)) > 1e5));
    
%     if plot_it
%         figure(10); subplot(2,1,1);plot(cur_stim(search_wind)); hold on;
%         plot(surround_wind(1),cur_stim(search_wind(surround_wind(1))),'g*');
%         plot((start_pulse_idx),cur_stim(search_wind(start_pulse_idx)),'r*');
%         % subplot(2,1,2); plot(2:(length(stim_reference)),kaiser_fact_stim)
%         
%         subplot(2,1,2); plot(cur_spiking(search_wind));
%         hold on; plot(surround_wind(1),cur_spiking(search_wind(surround_wind(1))),'*')
%         plot((start_pulse_idx ),cur_spiking(search_wind(start_pulse_idx)),'r*');
%     end
    [max_val center_idx] = max(cur_spiking(search_wind));
    pulse_start(n_p_idx) =search_wind(start_pulse_idx);
    max_spk_height(n_p_idx) = max_val;
end
%% Spontanoues spike locating:
fs = 1/min(unique(diff(cur_timing(1:10))));
filt_band = 500; %window size
spk_only_aff = highpass(cur_spiking,filt_band,fs);

wind_same_spk = 20;%.002;% time where say it's part of the same thing (s)
%Detect spike-y times
clear kaiser_fact
for n_step =2:(length(cur_spiking)-1)
    kaiser_fact(n_step) = cur_spiking(n_step)^2 - cur_spiking(n_step - 1) - cur_spiking(n_step +1);
end

% %Check kaiser spike detection:
if plot_it
    figure(9);
    ax1= subplot(3,1,1); plot(cur_timing,cur_stim,'k'); hold on;
    title('Stimulation Recording')
    ax2= subplot(3,1,2); plot(cur_timing,cur_spiking,'k'); hold on;
    title('Response Recording')
    ax3= subplot(3,1,3); plot(cur_timing(1:end-1),kaiser_fact,'k'); hold on;
    title('Kaiser Factor (with thresholding)')
    plot([0 cur_timing(end)],[k_thresh_min_max(1) k_thresh_min_max(1)],'r');
    plot([0 cur_timing(end)],[k_thresh_min_max(2) k_thresh_min_max(2)],'r');
    linkaxes([ax1 ax2 ax3],'x')
end

%Make sure they are spike shaped:
prob_spont_spk = [0 (kaiser_fact < k_thresh_min_max(2)) & (kaiser_fact > k_thresh_min_max(1))];
prob_spont_spk1 = [ prob_spont_spk & (cur_spiking >= 0)'];
spk_times_tmp = diff(find(prob_spont_spk1));
one_less_than_elim = find(spk_times_tmp < wind_same_spk);
spk_time_tmp_2 = find(prob_spont_spk1);
spk_time_tmp_2([one_less_than_elim+1]) = [];

%See if not counting in pulse windows (use same function): set to zero so
%just looking at windows
thresh_spk = 0;
min_inter_event_steps = pulse_art_wind; %aff1 - 50;
[spont_idxs] = detect_same_event(spk_time_tmp_2,pulse_start,cur_timing, thresh_spk,min_inter_event_steps);

% Test if spontaneous spikes are isolated:
%Looks strong on afferent 1. 8/2/21
if plot_it
    figure(3); plot(cur_timing, cur_spiking,'k'); hold on;
    plot(cur_timing(pulse_start), cur_spiking(pulse_start),'m*')
    plot(cur_timing(pulse_start+min_inter_event_steps), cur_spiking(pulse_start+min_inter_event_steps),'mo')
    plot(cur_timing(spk_time_tmp_2), cur_spiking(spk_time_tmp_2),'r.')
    plot(cur_timing(spont_idxs), cur_spiking(spont_idxs),'b.')
end

%% Identify pulses with spiking after or not:
post_pulse_window = 100; % for aff1?
% figure(4); plot(cur_timing, cur_spiking,'k'); hold on;
%  plot(cur_timing(pulse_start), cur_spiking(pulse_start),'r.'); hold on;

clear post_pulse_resp
for n_ps = 1:length(pulse_start)
    %plot(cur_spiking(pulse_start(n_ps):(pulse_start(n_ps)+post_pulse_window)),'k'); hold on;
    post_pulse_resp(n_ps,:) = cur_spiking((pulse_start(n_ps)):(pulse_start(n_ps)+post_pulse_window));
end

%Normaize and zero mean the data:
zmean_resp = (post_pulse_resp - mean(post_pulse_resp,2));
norm2_p_resp = zmean_resp./(max(zmean_resp,[],2) - min(zmean_resp,[],2));
clear post_pulse_resp

% %Try again to align to peak: (important especially for afferent 7)
[max_val idx_max]=max( norm2_p_resp,[],2);
shift = max(unique(idx_max));
back_step = min(idx_max) - 1;

euc_norm = @(x) sqrt(sum(x.^2,2));

clear norm2_pulse_resp
clear post_pulse_cos_sim
for n_pulses = 1:length(idx_max)
    off_end = (shift - idx_max(n_pulses));
    %post_pulse_resp_new
    norm2_pulse_resp(n_pulses,:) = norm2_p_resp(n_pulses,(idx_max(n_pulses)-back_step):end-off_end)./max_val(n_pulses);
end
for n_pulses = 1:length(norm2_pulse_resp)
    %Try cosine similarity:
    tmp = dot(repmat( norm2_pulse_resp(n_pulses,:),[size( norm2_pulse_resp,1) 1]), norm2_pulse_resp,2);
    tmp2 = tmp./(euc_norm( norm2_pulse_resp(n_pulses,:)).*(euc_norm( norm2_pulse_resp)));
    post_pulse_cos_sim(n_pulses,:) = tmp2;
end

% %Find most similar pulse responses:
% figure(7); subplot(2,1,1);plot( norm2_pulse_resp');
% hold on;

most_common_shape = sum((post_pulse_cos_sim > 0.99));
[val_max idx_max] = max(most_common_shape);
%Chose max comon shape as the reference shape:
ref_shape = norm2_pulse_resp(idx_max,:);

%%% Knee point algorithm on cosine symmetry seems to work best
%By knee point algorithm would need to

[sim_ord sim_idxs] = sort(post_pulse_cos_sim(idx_max,:),'ascend')
ordered_cos_sims = post_pulse_cos_sim(idx_max,sim_idxs);

sim_cutoff_idx = max(find(ordered_cos_sims < 0.99));


%Implement a knee point (Simply):
%Always ordered towards maximum value on the right
raw_diff= diff(ordered_cos_sims);
avg_step = 31;
n_cnt =1;
clear mov_std_diff
n_cnt =1;
x_axis_vals =  1:15:length(raw_diff)-(avg_step);
for n_step = x_axis_vals
    mov_std_diff(n_cnt) = std(raw_diff(n_step :(n_step +avg_step)));
    n_cnt =n_cnt +1;
end
%Make sure not very much off the stable point:
stable_thresh = 1e-05;

%Correct if there is a second knee - find that first one in continue
%round of values close to zero:
close_to_stable = find(mov_std_diff < stable_thresh);
settle_idx = max(find(diff(close_to_stable) ~= 1));

cut_off_idx = x_axis_vals(close_to_stable(settle_idx+1));
if plot_it
    %Plot to verify cutoff:
    figure(40); subplot(2,1,1);
    plot(raw_diff,'k'); hold on;
    subplot(2,1,2);
    plot(x_axis_vals,mov_std_diff,'r'); hold on;
    plot(x_axis_vals(mov_std_diff < stable_thresh), mov_std_diff(mov_std_diff < stable_thresh),'g.');
    plot(x_axis_vals(close_to_stable(settle_idx+1)), mov_std_diff(close_to_stable(settle_idx+1)),'m*');
end
% plot(x_axis_vals(1:end-1),diff(mov_std_diff),'b')
%mov_avg_diff= raw_diff;
%     p = polyfit(1:length( mov_avg_diff), mov_avg_diff,10)
%     %p2 = polyfit(1:length(raw_diff),raw_diff,8)
%     figure(10);ax1=
% subplot(2,1,1);
% plot(ordered_cos_sims,'k'); hold on;
%plot(polyval(p2,1:length( raw_diff)),'r','linewidth',2)
%     plot(x_axis_vals,  mov_avg_diff,'color',[0 0 0 .5],'linewidth',2);
%     hold on; plot(ceil(avg_step/2):7:length(raw_diff)-(avg_step/2),polyval(p,1:length(mov_avg_diff)),'b','linewidth',2);
%     ax2= subplot(2,1,2);
%     plot(ceil(avg_step/2):7:length(raw_diff)-(avg_step),diff( mov_avg_diff),'m');%polyval(p,1:length(mov_avg_diff))),'m');
%     linkaxes([ax1 ax2],'x')

figure(fig_num);
subplot(2,1,1);plot((ordered_cos_sims)); hold on;
plot(cut_off_idx,ordered_cos_sims(cut_off_idx),'r*');
subplot(2,2,3); plot(norm2_pulse_resp(sim_idxs(1:cut_off_idx),:)'); hold on;
title('Spike Pulse Trials');
subplot(2,2,4); plot(norm2_pulse_resp(sim_idxs((cut_off_idx+1):end),:)'); hold on;
title('No spike Pulse Trials')



spike_info.pulses_tot = pulse_start;
spike_info.pulse_evoked_spk_idxs = sort(pulse_start(sim_idxs(1: cut_off_idx )));
spike_info.spont_spk_idxs = spont_idxs;
%spike_info.pulse_evoked_spk_ts = cur_timing(pulse_start(sim_idxs(1:shape_cutoff)));

end
% figure(4);
% for cur_resp = 1:1000
% plot(norm2_pulse_resp(sim_idxs(cur_resp),:));
% title(num2str(sim_idxs(cur_resp)))
% pause;
% cla
% end
%
% figure(5);
% for cur_resp = 1001:length(sim_idxs)
% plot(norm2_pulse_resp(sim_idxs(cur_resp),:));
% title([num2str(sim_idxs(cur_resp)) ' ' num2str(cur_resp)])
% pause;
% cla
% end

%%
% %Tried subtraction: 8/2/31
% clear post_pulse_diff
% for n_post = 1:size(post_pulse_resp_new)
%     post_pulse_diff(n_post,:) = rms(post_pulse_resp_new(n_post,:) - post_pulse_resp_new,2);
%     post_pulse_diff(n_post,n_post) = nan;
% end
% [diff_fin sort_ord ] = sort(nansum(post_pulse_diff,2));
% [a sort_ord] = min(min(post_pulse_diff))
% min_e_resp = post_pulse_resp_new(sort_ord(1),:);
%processed_resp =(post_pulse_resp_new - min_e_resp);
% processed_resp =(norm2_pulse_resp - ref_shape);
%
%  plot(ref_shape,'k','linewidth',2)
% subplot(2,1,2);
% plot(processed_resp')

%% Principle Component Analysis
%pca_categorize(processed_resp)
%subplot(3,1,1);plot(normed_post_pulse_resp');
% % Look where outliers are (Afferent 1)
% %
% % idxs_1 = [414 1994];
% % idxs_2 = [107 108 137 230 278];
% % subplot(3,1,1);plot3(newdata(idxs_1,1),newdata(idxs_1,2),newdata(idxs_1,3),'mo')
% % subplot(3,1,1);plot3(newdata(idxs_2,1),newdata(idxs_2,2),newdata(idxs_2,3),'go')
% %  clust_idxs = find(km_cluster_labels == 3);
%  figure(10);
% for n_trace = idxs_2
%     plot(post_pulse_resp(n_trace,:));
%     title(num2str((n_trace)))
%     pause;
% end
