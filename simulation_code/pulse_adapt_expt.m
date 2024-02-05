function [firing block_ts curr_amps run_full] = pulse_adapt_expt(sim_info,pulse_rate,curr_options, change_param, tot_reps, output)
%Pulsatile Adaptation Experiment - determining if simulation replicates
%real data from Mitchell et al experiment with no vestibular afferent
%adaptation to multiple blocks of pulsatile stimulation after proceeding
%blocks of pulsatile stimulation at 100, 200, 300 pps.

%Sets up all the parameters for the experiment in preparation for running
%then runs in last lines

%In the mitchell experiment - "go through every pulse block then do 100
%pps, block, 200 pps, block, 300 pps, block - compare the blocks:
%" During test blocks, each pulse train lasted 1 s and rates of 25-300 p.p.s. were used.
%During activation blocks,  which consisted of 30 pulse trains lasting
%500?ms and were delivered every 2?s over the course of 1?min at pulse
%rates of 100pps, 200pps and 300pps. Test stimuli were then delivered every 2
%minutes for up to 10 minutes following vestibular nerve activation with 300pps."
%This is the slower implementation of this experiment because found running
%the individual X1 pps - 100 pps - X1 pps, X2 - 100 pps X2 pps, etc. shorter experiments gives same result
%
%%%%%Found can do the equivalent in blocked experiments so perform as
%%%%%pulsatile non-Plan experiments instead. The code to do as one
%%%%%experiment is here though:
%Start Date 9/2020
%Last Updated 12/29/20 CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Using isPlan to show adaptation (like in Mitchell et al 2016 paper):
pulse_height = 1; % can later find and change to whatever
stim_interv = 150;%uS (from Mitchell)
change_params.I_st = [];
plot_steps = 0;
run_full = 0; % one long block or several parallel pulse -block -pulse runs

if run_full
    curr_amps = [];
    change_params.neuron = change_param.neuron;
    block_ts = []; I_st = [];
    for n_blocks = 1:size(pulse_rate,2)
        n_blocks
        if pulse_rate(n_blocks) == 0
            tmp_I_st = zeros(1,1e6); % each pulse rate block is 1 second
            start_pt = 1;
        else

            pulse_times = round((1:1e6*(1/pulse_rate(n_blocks)):(1e6)));
            start_pt = length(I_st) + .5e6;%ms %500 ms between blocks - number not found in experiment*****!!!
            tmp_I_st = zeros(1,1e6);
            for n_ps = 1:length(pulse_times)
                tmp_I_st([round([pulse_times(n_ps):pulse_times(n_ps)+stim_interv-1])]) = pulse_height;
                tmp_I_st([round([(pulse_times(n_ps)+stim_interv):(pulse_times(n_ps)+2*stim_interv-1)])]) = -pulse_height;
            end
        end
        I_st = [I_st zeros(1,.5e6) tmp_I_st];
        change_units = 'ms';
        block_ts = [block_ts start_pt length(I_st)];%[block_ts start_pt length(I_st)];
    end

    % In real experiment need blocks of each could just run with one amplitude
    % surrounded by the two sessions instead though
    p_amps = [100 200 300];
    active_block = zeros(1,500*1e3); % each activation block is 500 ms
    p_times = round(1e3*(1:((1/p_amps(1))*1e3):(500))); % 500 ms in mitchell
    for n_ps = 1:length(p_times)
        active_block([round([p_times(n_ps):p_times(n_ps)+stim_interv-1])]) = pulse_height;
        active_block([round([(p_times(n_ps)+stim_interv):(p_times(n_ps)+2*stim_interv-1)])]) = -pulse_height;
    end

    num_a_blocks=10;% repeat activation blokcs 10 times (really 30 in experiment)
    act_block_tot= repmat([active_block zeros(1,1500e3)],[1 num_a_blocks]); % calcualted so that active + stim comes to 2 seconds
    block_ts = [block_ts block_ts(end)+.5e6+[0 length(act_block_tot) length(act_block_tot)+block_ts]];
    fin_I_st = [I_st zeros(1,.5e6) act_block_tot I_st];
    I_st = fin_I_st;
    clear fin_I_st;
    sim_info.sim_time = (length(I_st)/1e3);

    %%%%%%%%%
    firing = struct();
    for n_currents = 1:length(curr_options)
        %  change_params =struct();
        change_params.I_st = I_st;
        idx_plus = find(change_params.I_st == 1);
        idx_minus = find(change_params.I_st == -1);

        change_params.I_st(idx_plus) = curr_options(n_currents);
        change_params.I_st(idx_minus) = -curr_options(n_currents);

        change_params.Inj_cur = curr_options(n_currents);
        change_params.pulse_rate = [];

        % disp(sprintf('%d of %d, give %s  %s',n_experiments, length(changing),...
        %     num2str(changing(n_experiments)),change_units))

        %Set up different cases of running the parfor for different
        %experiments that require different variables

        %%%%%%%% Run Simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for rep_num = 1:tot_reps
            %%%   rng(rep_num); - if want to try setting seed and comparing
            [spiking_info] =  run_expt_on_axon_10_10_20_f(sim_info,output,change_params)
            %single_node_KH_adapt_v1_f(sim_info,output,change_params);
            firing(n_currents).rep(rep_num).times = spiking_info.end.spk_times;
        end
    end

else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Run for 100 center block, 200 center block, 300 center block in
    %parallel
    % Faster computer, easier if want to try at different current
    % amplitudes

    % In real experiment need blocks of each could just run with one amplitude
    % surrounded by the two sessions instead though
    firing = struct();

    p_blck_len = 1e6;
    active_len = 1000; %ms
    active_block = zeros(1,active_len*1e3);
    t_btw_blocks = .5e6;
    start_shift = sim_info.sim_start_time*1e3; %1e-3 ms units
    block_ts = start_shift  +[0 p_blck_len p_blck_len+t_btw_blocks p_blck_len+t_btw_blocks+length(active_block)...
        p_blck_len+2*t_btw_blocks+length(active_block) 2*p_blck_len+2*t_btw_blocks+length(active_block)];

    sim_info.sim_time = (block_ts(end))/1e3;


    for n_currents = 1:length(curr_options)
        % Each Central Amplitude
        p_amps = [100 200 300];

        %par
        parfor n_cntr_amp = 1:length(p_amps)
            active_len = 1000; %ms
            active_block = zeros(1,active_len*1e3); % each activation block is 500 ms
            p_times = round(1e3*(1:((1/p_amps(n_cntr_amp))*1e3):(active_len))); % 500 ms in mitchell

            for n_ps = 1:length(p_times)
                active_block([round([p_times(n_ps):p_times(n_ps)+stim_interv-1])]) = pulse_height;
                active_block([round([(p_times(n_ps)+stim_interv):(p_times(n_ps)+2*stim_interv-1)])]) = -pulse_height;
            end

            %Each Pulse Rate
            spk_per_bin = zeros(length(pulse_rate), tot_reps,3);

            pr = struct();
            for n_prs = 1:length(pulse_rate)

                t_btw_blocks = .5e6;%ms %500 ms between blocks - number not found in experiment*****!!!
                p_blck_len = 1e6;
                if pulse_rate(n_prs) == 0

                    tmp_I_st = zeros(1,p_blck_len); % each pulse rate block is 1 second
                else
                    pulse_times = round((1:1e6*(1/pulse_rate(n_prs)):(p_blck_len)));

                    tmp_I_st = zeros(1,p_blck_len);
                    for n_ps = 1:length(pulse_times)
                        tmp_I_st([round([pulse_times(n_ps):pulse_times(n_ps)+stim_interv-1])]) = pulse_height;
                        tmp_I_st([round([(pulse_times(n_ps)+stim_interv):(pulse_times(n_ps)+2*stim_interv-1)])]) = -pulse_height;
                    end
                end

                I_st_full = [zeros(1,start_shift), tmp_I_st, zeros(1,t_btw_blocks), active_block, zeros(1,t_btw_blocks), tmp_I_st];
                change_params = struct();
                change_params.I_st = I_st_full;
                idx_plus = find(change_params.I_st == 1);
                idx_minus = find(change_params.I_st == -1);

                change_params.I_st(idx_plus) = curr_options(n_currents);
                change_params.I_st(idx_minus) = -curr_options(n_currents);
                change_params.Inj_cur = curr_options;
                change_params.neuron = change_param.neuron;
                change_params.pulse_rate = pulse_rate(n_prs);
                %%%%%%%% Run Simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                spk_per_tmp = zeros(tot_reps, 3);
                for rep_num = 1:tot_reps
                    %%%   rng(rep_num); - if want to try setting seed and comparing
                    [spiking_info] =  run_expt_on_axon_10_10_20_f(sim_info,output,change_params);
                    %single_node_KH_adapt_v1_f(sim_info,output,change_params);
                    pr(n_prs).rep(rep_num).times = spiking_info.end.spk_times;
                    spike_ts = spiking_info.end.spk_times;
                    n_cnt =1;
                    for n_pulse_trials = 2:2:length(block_ts)
                        spk_per_tmp(rep_num,n_cnt) = sum((spike_ts < block_ts(n_pulse_trials)/1e3) &  ...
                            (spike_ts >  block_ts(n_pulse_trials-1)/1e3))/((block_ts(n_pulse_trials)-block_ts(n_pulse_trials-1))/1e6);
                        n_cnt = n_cnt +1;
                    end
                end

                spk_per_bin(n_prs,:,:) = spk_per_tmp;

                if plot_steps % demo results:
                    figure(1);
                    subplot(2,1,1);
                    plot(I_st_full); hold on; ylabel('Pulse Timing')
                    plot(block_ts,ones(size(block_ts)),'.')
                    subplot(2,1,2);
                    plot(spiking_info.end.spk_times,ones(size(spiking_info.end.spk_times)),'.');

                    figure(10); plot([1 2 3],squeeze(spk_per_bin(n_prs,:,:)),'.--')
                end
            end

            blck_amp(n_cntr_amp).spk_per_bin = spk_per_bin;

        end
        curr_amps(n_currents).blck_cur  = p_amps;
        curr_amps(n_currents).diff_blck_prs = blck_amp;

    end

    firing = [];
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions for above
%%% For designing biphasic pusle based experiments
function [I_st] = make_bp_pulses(cur_timing,sess_length,I_inj)
stim_interval_length = 150;
I_st = zeros(1,sess_length);
for num_ps = 1:length(cur_timing)
    I_st(cur_timing(num_ps):cur_timing(num_ps)+stim_interval_length-1) = I_inj;%7.5; %uA - value (p 65)
    I_st(cur_timing(num_ps)+stim_interval_length:cur_timing(num_ps)+round(2*stim_interval_length)-1) = -I_inj;
end
end

function [I_st,session_start_end] = add_test_sess(I_st,cur_sesses,p_rate,sess_time,I_inj)
for n_prates = 1:length(p_rate)
    session_start_end(cur_sesses+n_prates,1) = (length(I_st)+1)/1e3;
    interp_interv_dt = (1/p_rate(n_prates))*1e6;
    cur_timing = round(1:interp_interv_dt:sess_time*1e3);
    [temp_sess] = make_bp_pulses(cur_timing,sess_time*1e3,I_inj);
    I_st = [I_st temp_sess];
    session_start_end(cur_sesses+n_prates,2) = 1e-3*length(I_st);
    if n_prates < length(p_rate)
        I_st = [I_st zeros(1,500*1e3)];
    end
end
end