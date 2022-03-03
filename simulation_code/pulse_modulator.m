function [firing sim_info] = pulse_modulator(sim_info,input_mod_f,curr_options, change_params, tot_reps, output,rate_mode)
%Pulse rate modulation code
%Takes in a function and modulates pulse rate with shape of function
%https://pubmed.ncbi.nlm.nih.gov/21859631/ - every 10 ms change sampling of
%function: .5  to 10 Hz sinuosoid.
%Started 12/27/20 CRS
%Last Updated 12/27/20 CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%rate_mode =1;%amp_modulate = 0
plot_steps = 0; %plot graphs to check along the way
dt = 1e-6; %s
if isempty(input_mod_f.mod_f)
    
    sim_time = round(2+sim_info.sim_start_time*1e-3); %s
    sim_info.sim_time = sim_time/1e-3;
    pulse_height = curr_options;
    t_full = 0:dt:sim_time-dt; %s
    mean_val = 175;
    mod_amp = 50*[1 1];%pr
    %scale = [1 1]%[1 5.5];%[1 5.5]; %neg, pos
    mod_freq = 10; %Hz % vary with sins from 0 - 10
    delay = 500; %ms
    
    mod_function = sin(mod_freq*2*pi*t_full);
    timing = [zeros(1,.5*delay*1e3) -ones(1,.5*delay*1e3) mod_function([1:(1e3*(sim_time*1e3-delay))])];
    mod_function(mod_function > 0)= mod_function(mod_function > 0)*mod_amp;
    mod_function(mod_function < 0)= mod_function(mod_function < 0)*mod_amp;
    mod_function = [zeros(1,.5*delay*1e3) -mean_val*ones(1,.5*delay*1e3) mod_function([1:(1e3*(sim_time*1e3-delay))])];
    mod_function = mod_function+ mean_val;
    firing.delay = delay;
    firing.mod_f = mod_function;
    firing.mod_timing = timing;
else
    mean_val = input_mod_f.pm_base;
    mod_amp = input_mod_f.pm_mod_amp;
    sim_time = input_mod_f.sim_time;
    mod_function = input_mod_f.mod_f;
    t_full = input_mod_f.mod_timing;
    firing = input_mod_f;
    delay = 400; %ms
    pulse_height = curr_options;
    dt_t_full = diff(t_full(1:2));
end


update_freq= 1*1e-4;%s%10*1e-3;%s
smple_freq = update_freq/dt_t_full;

%Make Pulse over Time
stim_interv = 150;%uS (from Mitchell)

if rate_mode
    %tried to assure for taking any fr tranjectory has same dt
    [p_times]  = mod_f_t_pr(mod_function,smple_freq,t_full,dt, plot_steps);
        
    pulse_times = round(p_times+delay*1e3+sim_info.sim_start_time*1e3);
    pulse_times = pulse_times(pulse_times < round(sim_time*1e6));
    
    I_st = zeros(1,round(sim_time/dt));
    for n_ps = 1:length(pulse_times)
        I_st([round([pulse_times(n_ps):pulse_times(n_ps)+stim_interv-1])]) = pulse_height;
        I_st([round([(pulse_times(n_ps)+stim_interv):(pulse_times(n_ps)+2*stim_interv-1)])]) = -pulse_height;
    end
    if plot_steps
        figure(2);
        ax1=subplot(2,1,1);plot(t_full+delay*1e-3+sim_info.sim_start_time*1e-3, mod_function); xlim([0 sim_time])
        ax2=subplot(2,1,2); plot(dt*(1:length(I_st)),I_st); xlabel('Time (s)'); ylabel('I_{inj}');
        xlim([0 sim_time])
        linkaxes([ax1 ax2],'x')
    end
    
    
    %Variables for Simulation
    I_st = I_st(1:round(sim_time/dt));
    change_params.I_st = I_st;
    change_params.Inj_cur = curr_options;
    sim_info.isPlan = 1 ;
    for rep_num = 1:tot_reps
        %%%   rng(rep_num); - if want to try setting seed and comparing
        %Run it right here:
        [spiking_info] =  run_expt_on_axon_10_10_20_f(sim_info,output,change_params);
        firing.rep(rep_num).times = spiking_info.end.spk_times;
    end
    firing.rep(rep_num).pulse_times = pulse_times;
    firing.mod_freq= input_mod_f.mod_freq;
    firing.rep(rep_num).times = spiking_info.end.spk_times;
    
else
    pr = firing.pm_base; %pps
    p_times = 1:round((1e3/pr)*1e3):sim_time*1e6;
    pulse_times = round(p_times+delay*1e3+sim_info.sim_start_time*1e3);
    pulse_times = pulse_times(pulse_times < round(sim_time*1e6-2*stim_interv));
    if firing.set_f
        mod_function = input_mod_f.mod_f/-20;
    else
    %Amplitude Modulation:
    I_base = curr_options;%*-20; %what pulsatile stimulation tracel looks like
    %mod_depth = .3;
    mod_range_mag = input_mod_f.pm_mod_amp/-20;% [7] %100-200 uA
    mod_function = (mod_range_mag.*sin(input_mod_f.mod_freq*2*pi*t_full)) + I_base;
    end
     
    I_st = zeros(1,sim_time/dt);
    for n_ps = 1:length(pulse_times) 
        I_st([round([pulse_times(n_ps):pulse_times(n_ps)+stim_interv-1])]) = mod_function(max(round(p_times(n_ps)*1e-2),1));
        I_st([round([(pulse_times(n_ps)+stim_interv):(pulse_times(n_ps)+2*stim_interv-1)])]) =  -mod_function(max(round(p_times(n_ps)*1e-2),1));
    end
     
    firing.mod_f = mod_function;
    if plot_steps
    figure(2);
    ax1=subplot(2,1,1);plot(t_full+delay*1e-3+sim_info.sim_start_time*1e-3, mod_function); xlim([0 sim_time])
    ax2=subplot(2,1,2); plot(dt*(1:length(I_st)),I_st); xlabel('Time (s)'); ylabel('I_{inj}');
    xlim([0 sim_time])
    linkaxes([ax1 ax2],'x')
     end   
    change_params.I_st = I_st;
    %change_params.I_base= I_base;
    %change_params.I_mod = mod_depth;
    change_params.Inj_cur = curr_options;
    sim_info.isPlan =1 ;
    for rep_num = 1:tot_reps
        %%%   rng(rep_num); - if want to try setting seed and comparing
        [spiking_info] =  run_expt_on_axon_10_10_20_f(sim_info,output,change_params);
        firing.rep(rep_num).times = spiking_info.end.spk_times;
    end
    firing.rep(rep_num).pulse_times = pulse_times;
end
firing.fin_t = t_full+delay*1e-3+sim_info.sim_start_time*1e-3;
firing.fin_m = mod_function;
firing.I_st = I_st;
firing.I_timing = dt*(1:length(I_st));
end


function [p_times] = mod_f_t_pr(mod_function,smple_freq,t,dt, plot_steps)
t_mod_smple = 1:(smple_freq):length(t);
t_lst_pulse = 0;
p_times = [1];
next_p_times = nan;
%Change  -do not hadd pulse until the cur_t is PAST IT in iteration
for cur_t = 1:(length(t_mod_smple)-1)
    %At this time what is the new pulse rate
    cur_pr = mod_function(t_mod_smple(cur_t));
    smple_freq_in_dt = (smple_freq*(t(2)-t(1)))/dt;
    if isinf(cur_pr)
        cur_ipi = 1e12;
    else
        cur_ipi = round((1/cur_pr)/dt); %s - dt adjust dt (us)
    end
    
    %Case when there has been a time where there has been no pulses in a
    %very long time:
    if  (t(t_mod_smple(cur_t))/dt - t_lst_pulse) > .5/dt % in dt 
        %^ upper limit capping this rule
        disp('Too slow')
        t_lst_pulse = t(t_mod_smple(cur_t))/dt;
        if cur_pr > 0
            p_times = [p_times t_lst_pulse];% in dt us
        end
    end
    
    %Give pulses at current frequency until next sample
    %Finds last pulse in sequence
    if cur_ipi < smple_freq_in_dt % in us
        disp('Too fast')
        next_p_times = t_lst_pulse:cur_ipi:(t_lst_pulse+smple_freq_in_dt);
        next_p_times(next_p_times == t_lst_pulse) = [];
        t_lst_pulse = round(next_p_times(end));
        p_times = [p_times round(next_p_times)];% in dt/us
        
    else
        %Always the predicted next time:
         %If the current time is less than the predicted next pulse time
        %don't update the time of next pulse arrival yet:
        cur_pred_next_p_time = min(t_lst_pulse + cur_ipi,t(t_mod_smple(end))/dt);
    
        if ((t(t_mod_smple(cur_t))/dt < next_p_times) ) | (isnan(next_p_times))
            %don't update the last pulse or prediction time until get to
            %last predicted pulse (within reason - carried out above) 
            %if (cur_pred_next_p_time < .75*next_p_times)
            next_p_times = cur_pred_next_p_time; %The up to date prediction of next pulse time
            %end    
        else
            if (t(t_mod_smple(cur_t))/dt >= next_p_times) 
                p_times = [p_times round(next_p_times)];
                t_lst_pulse = round(next_p_times);
                next_p_times = t_lst_pulse + cur_ipi;
            end
        %So otherwise t_lst_pulse is the same
        end
       
    end
end

if plot_steps
    figure(1); ax1= subplot(2,1,1);
    plot(1e3*t,mod_function,'.'); hold on;
    plot(1e3*t(t_mod_smple),mod_function(t_mod_smple),'o'); box off;
    title('Modulation function'); ylabel('Pulse Rate (pps)')
    ax2=subplot(2,1,2);
    plot(1e3*p_times*dt,ones(size(p_times)),'o'); hold on; box off;
    plot(1e3*t(t_mod_smple), ones(size(t_mod_smple)),'.');
    xlabel('Time (ms)'); ylabel('Pulse Timing')
    linkaxes([ax1 ax2 ],'x')
end
end
%%