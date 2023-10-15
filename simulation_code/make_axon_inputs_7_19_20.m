function [axon_inputs change_params] = make_axon_inputs_7_19_20(change_params, sim_info)
Cm = 0.9; % uF/cm2 - capacitance per SA
S = 10e-6/Cm;%pi*d*l; % becuase 10 pF total
rho_i = .7*1e2;%.7*1e2;%.7*1e2;%ohm*m - internal resistivity
rho_e = 3*1e2;%ohm*m --> Ohm*cm - external resistivity
%keep standard in all cases for now
dt = 0.001;%ms - time step of whole simulation
%Starting from assuming one node and s1 shaped EPSCs
which_s = 1; % choose between s1,s2,s3

if sim_info.inj_cur(1)
    Inj_current = change_params.Inj_cur; %2e-2;%.00001;%[10:20:140];
else
    Inj_current = 0;
end
if sim_info.isPlan
%For Pulsatile Stimulation:
       if change_params.full_seq %For introducing a particularly shaped pulse instead of just specifically timed pulses (added 7/22 after jitter analysis)
           change_params.I_st = sim_info.I_st;%put the whole thing into  sim_info at the beginning - would only be in this full trace plan
       else
        tmp_timing = (find( change_params.I_st ~=0));
        if ~isempty(tmp_timing)
            pulse_timing = tmp_timing([1 find(diff(tmp_timing) >1)]);
        else
            pulse_timing = [];
        end
        change_params.pulse_timing = pulse_timing;
       end
else
    I_st = zeros(1,sim_info.sim_time/dt);
    if sim_info.isDC
        if isempty(sim_info.sim_end_time)
            I_st(round(sim_info.sim_start_time/dt):round(sim_info.sim_time/dt)-1) = Inj_current;
        else
            I_st(round(sim_info.sim_start_time/dt):round(sim_info.sim_end_time/dt)) = Inj_current;
        end
        
    else
        if change_params.pulse_rate == 0
            pulse_timing = [];
        else
            %For Pulsatile Stimulation:
            stim_interval_length = 150;%us = 1e-3 ms
            inter_pulse_interval = round(1e6/change_params.pulse_rate); %1 second in dt then pulse rate
            pulse_timing = [round(sim_info.sim_start_time/dt):inter_pulse_interval:round(sim_info.sim_time/dt)];
            
            for num_ps = 1:length(pulse_timing)
                I_st(pulse_timing(num_ps):pulse_timing(num_ps)+stim_interval_length-1) = Inj_current;%7.5; %uA - value (p 65)
                I_st(pulse_timing(num_ps)+stim_interval_length:pulse_timing(num_ps)+round(2*stim_interval_length)-1) = -Inj_current;
            end
            spiking_info.pulse= round(pulse_timing*1e-3)+1;
        end
        change_params.pulse_timing = pulse_timing;
    end
    change_params.I_st = I_st(1:round(sim_info.sim_time/dt)); I_st = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Simulate current fields
if sim_info.inj_cur(1)
    adapt_input = change_params.I_st;
% %         vert_dist =.2e-1; %.05e-1;%%cm = 1 mm from axon %initial distance
% %         X_tot =  0;
        vert_dist =.2;%%.2; %.1 cm = 1 mm from axon %initial distance       
        X_tot =  0.7;%0.7
        elec_dist_tot = sqrt(X_tot.^2 + vert_dist.^2)'; %cm
        stim_I = Inj_current; % -0.1;%mA? - Blocks (can check lower)
        %Try to reconstruct from Frijn and here:
        %pi*d*l  d ~=10l - Frijns , d~= 2l - vestibular papers3-5 v, 2-5
        dl_Ratio = 9;%3;%6/2;% 2; from sources p61 labnotebook
        d = sqrt(S/dl_Ratio*pi)*dl_Ratio;%sqrt(S/(11*pi))*10;%S = pi*d*l assume d ~= 3l S ~=4pil^2
        L = (100/(0.7))*d;
       
       new_scale = (-(S./(4*pi.*elec_dist_tot^2)));%*3.7^2)));
       old_scale = (-(pi*d^2/(4*L*rho_i))*(rho_e/(4*pi))/(.1*elec_dist_tot));
    %%%%   (new_scale*1000)/old_scale
       if  sim_info.isDC
            
            I_stim =  new_scale.*change_params.I_st*(1000);%/13);
            %%new_scale.*change_params.I_st;

% %      %   if  sim_info.isDC
% %             
% %      %       V_stim = -(pi*d^2/(4*L*rho_i))*(rho_e/(4*pi))*(change_params.I_st)./repmat(elec_dist_tot, ...
% %      %           [1 sim_info.sim_time/dt]);
            %         figure(1); plot(V_stim); hold on;
            if sum(I_stim) ~= 0%sum(V_stim > 0) > 0
                %Change when do more complicated DC stim sesison

                calyx_non_quant = sim_info.non_quant;%prev 2 %from Goldberg fig. 3 comparison of base fr shift hsould be same and 2 for both%%%(find(V_stim > 0)) =  2;%4;%.5;%4;
                I_stim = calyx_non_quant.*I_stim;
                change_params.I_st = calyx_non_quant.*change_params.I_st;           
            end
            
        else
            I_stim = new_scale.*change_params.I_st*(1000);
            %-(pi*d^2/(4*L*rho_i))*(rho_e/(4*pi))*(change_params.I_st)./repmat(elec_dist_tot, ...
            %    [1 sim_info.sim_time/dt]);
        end
else
    I_stim = zeros(1,sim_info.sim_time*1e3);
end

if sim_info.inj_cur(2)
    conv_fact = 1e-6;%pA --> uA
    if ~sim_info.epsc_over_ride %control exact epsc height/time
    
    sampling_rate = max(round(sim_info.mu_IPT*sim_info.epsc_sample),1); %max(round(sim_info.mu_IPT*5),1); %in ms == 1000*1e-3ms,  in 1 ms updates in EPSC statistics
    
 %   if   sim_info.isDC & (sum(change_params.I_st ~= 0) > 0)
      %  if sim_info.do_adapt_sens == 1
            %Funciton generally takes the curernt injects, converts to
            %excitation and calculates amount of adaptation
     if sim_info.do_adapt
%            if sim_info.cont_epsc
%           [adapt_function] = synapse_adapt_3_4_20_f_2(-adapt_input*1e3,1,sim_info,sim_info.do_orig);%sim_info.scale_sens,1);%use original equation
%           else
           [adapt_function] = synapse_adapt_5_28_20_f(-adapt_input*1e3,1,sim_info,sim_info.do_orig);%sim_info.scale_sens,1);%use original equation
%            end
     else
          adapt_function = zeros(1,round(length(change_params.I_st)/1e3));
     end
    
    
% if sim_info.cont_epsc
%      adapt_mu = sim_info.mu_IPT./max(0.002,(1+...
%             sim_info.mu_IPT.*sim_info.scale_adapt_mu*adapt_function));
%  
%     [I_epsc_tmp] = generate_IEPSC_2(sim_info.sim_time,which_s,adapt_mu,dt,0);
%     I_epsc = I_epsc_tmp*conv_fact*sim_info.curr_scaling;
% else
    
% if sim_info.isPlan & ~isempty(sim_info.stim_times)    
%     samp_num = [1:sampling_rate:sim_info.stim_times(1)-1 sim_info.stim_times(1):sampling_rate:length(adapt_function)];
% else
    samp_num = [1:sampling_rate:length(adapt_function)];
% end

    I_epsc= [];    
    for cur_section = 1:length(samp_num)        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Adaptation component
        adapt_mu = sim_info.mu_IPT./max(0.002,(1+...
            (1/sim_info.fr_o)*sim_info.scale_adapt_mu*adapt_function(round(samp_num(cur_section)))));
   %Instead of for sim_info.sim_time use only a ~sampling_Rate length
        %window
        I_epsc_tmp = generate_IEPSC(sampling_rate,change_params,which_s,adapt_mu,dt,sim_info.epsc_over_ride,0);
        I_epsc = [I_epsc conv_fact*I_epsc_tmp(1:round(sampling_rate/dt))];
    end
    else
         I_epsc= [];
          I_epsc_tmp = generate_IEPSC(length(change_params.I_st)/1e3,change_params,which_s,sim_info.mu_IPT,dt,sim_info.epsc_over_ride,0);
         I_epsc = [I_epsc conv_fact*I_epsc_tmp];

    end
    I_epsc = sim_info.epsc_scale*I_epsc;
%end

    %%%% Visualizing how adaptation relates to input, "excitation". and the
    %%%% I_epscs
    % %        subplot(5,1,1);
    % %         plot(adapt_input); hold on;
    % %         subplot(5,1,2);
    % %         plot(-adapt_input*1e3); hold on;
    % %         subplot(5,1,3);
    % %         plot(adapt_function); hold on;
    % %         subplot(5,1,4);
    % %           plot(full_adapt_seq);   hold on;
    % %      subplot(5,1,5);
    % %         clear fr
    % %         step =20;%3;
    % %         sec_samps = step/2+1:(length(samp_num)-step);
    % %         for n_sec = 1:length(sec_samps)-1
    % %            fr(n_sec) = sum(I_epsc([samp_num(sec_samps(n_sec)-step/2):samp_num(sec_samps(n_sec)+step/2)]*1e3) )  ;
    % %         end
    % %         plot(fr); hold on;
    
    %%% This is different than doing the above ...
    %Was originally just doing it all at once if not doing adaptation
    %To keep all fr the same do all I_epsc generation in stages
    %        I_epsc_tmp = generate_IEPSC(sim_info.sim_time,which_s,sim_info.mu_IPT,dt,0);
    %        I_epsc = sim_info.curr_scaling*conv_fact*I_epsc_tmp(1:round(sim_info.sim_time/dt));
    %adapt_function = ones(size(adapt_function));
    I_epsc = I_epsc(1:round(sim_info.sim_time/dt));
else
    I_epsc= zeros(size(I_stim));
end

axon_inputs.I_epsc = I_epsc;
axon_inputs.V_stim = I_stim; %(mA)
%axon_inputs.I_st = change_params.I_st; change_params.I_st = [];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function used above:
%Generate I_epscs in windows
function [I_epsc] = generate_IEPSC(sim_time,change_params,which_s,mu_IPT,dt,over_ride,plot_it)

        mu = 150; %pA nominal mean
        stdev= 115;%pA - std
        num_pulses = floor(sim_time/(mu_IPT*.03));
        pulse_heights = abs(normrnd(mu,stdev,[num_pulses,1]));
        %Replace and keep heights between 1.5 and 450: - read wrong
        num_replaces = length(find(pulse_heights > 450));
        need_rep = find(pulse_heights > 450);
        pulse_heights(need_rep) = abs(normrnd(mu,stdev,[num_replaces,1]))+1.5;
        
        %figure(1); subplot(2,1,1);histogram(pulse_heights);
        %Check values:[mean(pulse_heights) std(pulse_heights)]
        % Draw interpulse arrival times from exponential distribution
        %mu_IPT = 3; %ms %standard but was varied in the experiments
        %mu,a,b where get [a b] sized matrix
        IPTs = exprnd(mu_IPT,num_pulses,1);
        
% % %         figure(10); 
% % % %         subplot(2,1,1); histogram(pulse_heights); box off;
% % % %         xlabel('EPSC Amplitude (pA)'); ylabel('Number of EPSCs'); hold on;
% % % %         set(gca,'fontsize',16)
% % % %         subplot(2,1,2); 
% % %         %histogram(IPTs,25,'facecolor',[0.3010 0.7450 0.9330]);
% % %         %histogram(IPTs,25,'facecolor',[0 0 1]);%3
% % %         histogram(IPTs,25,'facecolor',[0.4940 0.1840 0.5560]);%0.75
% % %         xlabel('inter-EPSC interval (ms)'); ylabel('Number of EPSCs'); box off
% % %         set(gca,'fontsize',16); hold on;
% % %         xlim([0 25]); ylim([0 9000])
        
        %Because can now do less that ms interpulse interval not rounding:
        if over_ride
            epsc_times = change_params.epsc_time/dt;
            pulse_heights = change_params.epsc_height;
        else
         epsc_times = ceil(cumsum(IPTs)/dt); %in 1/1000 s of ms - ceil so never get zero
        end
        idx_p_ts = find(epsc_times < sim_time/dt);
        pulse_trains = zeros(1,round(sim_time/dt+sim_time*.2/dt)); %ms
        
        pulse_trains(round(epsc_times(idx_p_ts))) = pulse_heights(1:length(idx_p_ts));
        % To check pulse heights that lead to APs
        %pulse_times = 5000; % just one pulse
        %pulse_heights =sim_info.fake_EPSC_height;
        %pulse_trains(pulse_times) = pulse_heights;
        %Check values: mean(IPTs)
        %subplot(2,1,2); histogram(IPTs)
        %figure(1); plot([1:length(pulse_trains)]*dt,pulse_trains);
        %xlabel('time (ms)'); ylabel('EPSP heights')
        %EPSC shapes:
        t_epsc = 0:.001:15; %ms
        alpha_1= 0.4; %%%%%%s1
        s1 = (t_epsc./(alpha_1^2)).*exp(-t_epsc./alpha_1);
        alpha_2 = 0.4; %%%%%%s2
        s2 =   (t_epsc./(alpha_2^2)).*exp(-t_epsc./alpha_2);
        subset_t = t_epsc(find(t_epsc >= alpha_2));
        s2(find(t_epsc >= alpha_2)) = (subset_t./(alpha_2^2)).*exp(-subset_t./alpha_2) ...
            - exp(-(subset_t-alpha_2)/0.7) +0.8*exp(-(subset_t-alpha_2)/0.7)+ ...
            0.2*exp(-(subset_t-alpha_2)/3.2);
        alpha_3 = 4; %%%%%%s3
        s3 = (t_epsc./(alpha_3^2)).*exp(-t_epsc./alpha_3);
        %Not in paper but seems like need to normalize all to height max of 1.
        %figure(2); plot(t,s1); hold on; plot(t,s2); plot(t,s3); % Unnormalized
        s1= s1/max(s1); s2= s2/max(s2); s3= s3/max(s3);
        %figure(1); plot(t,s1/max(s1)); hold on; plot(t,s2/max(s2)); plot(t,s3/max(s3));
        if which_s ==1
            cur_shape = s1;
        else
            if which_s == 2
                cur_shape = s2;
            else
                cur_shape = s3;
            end
        end

        I_epsc = conv(pulse_trains,cur_shape);
        if plot_it
            figure(2); plot(dt*[1:length(I_epsc)],I_epsc);
            xlabel('time (ms)'); ylabel('EPSCs')
            figure(1); plot(I_epsc); hold on; plot([0 length(pulse_trains)],[237 237]); % show boundary to fire
        end
end

%Generate I_epscs continuously (not used - turned out NOT to be accurate)
function [I_epsc] = generate_IEPSC_2(sim_time,which_s,mu_IPT,dt,plot_it)
       
mu = 150; %pA nominal mean
stdev= 115;%pA - std

cur_t = 0;

pulse_trains = zeros(1,round(sim_time/dt)); %ms
while cur_t < sim_time/dt
    pulse_height = abs(normrnd(mu,stdev,1));
    while pulse_height < 0
        pulse_height = abs(normrnd(mu,stdev,1));
    end
    cur_mu = mu_IPT(min(max(cur_t,1),sim_time/dt));% mu_IPT(min(max(round(cur_t/1e3),1),sim_time));
  % [cur_t/1e3    cur_mu]
    IPTs = exprnd(cur_mu,1,1);
    cur_t = ceil(cur_t + IPTs*1e3);
    % In 1e-3 ms:
    pulse_trains(cur_t) = pulse_height;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EPSC shapes:
t_epsc = 0:.001:15; %ms
alpha_1= 0.4; %%%%%%s1
s1 = (t_epsc./(alpha_1^2)).*exp(-t_epsc./alpha_1);
alpha_2 = 0.4; %%%%%%s2
s2 =   (t_epsc./(alpha_2^2)).*exp(-t_epsc./alpha_2);
subset_t = t_epsc(find(t_epsc >= alpha_2));
s2(find(t_epsc >= alpha_2)) = (subset_t./(alpha_2^2)).*exp(-subset_t./alpha_2) ...
    - exp(-(subset_t-alpha_2)/0.7) +0.8*exp(-(subset_t-alpha_2)/0.7)+ ...
    0.2*exp(-(subset_t-alpha_2)/3.2);
alpha_3 = 4; %%%%%%s3
s3 = (t_epsc./(alpha_3^2)).*exp(-t_epsc./alpha_3);
s1= s1/max(s1); s2= s2/max(s2); s3= s3/max(s3);
%figure(1); plot(t,s1/max(s1)); hold on; plot(t,s2/max(s2)); plot(t,s3/max(s3));
if which_s ==1
    cur_shape = s1;
else
    if which_s == 2
        cur_shape = s2;
    else
        cur_shape = s3;
    end
end

I_epsc = conv(pulse_trains,cur_shape);

if plot_it
    %figure(2); plot(dt*[1:length(I_epsc)],I_epsc);
    
    figure(1); plot(dt*[1:length(I_epsc)],I_epsc); hold on;
    plot([0 dt*length(I_epsc)],[237 237]); % show boundary to fire
    xlabel('time (ms)'); ylabel('EPSCs')
    
    figure(2); subplot(2,1,1);
    plot([1:length(mu_IPT)]*dt,mu_IPT); xlabel('time'); ylabel('mu')
    subplot(2,1,2);
    
    bin_step = 50;
    clear m_std bin_ts
    bin_ts = [0:bin_step:sim_time]*1e3;
    
    spk_times = find(pulse_trains ~= 0);
    for n_bins = 1:length(bin_ts)-1
        m_std(n_bins,:) = [mean(diff(spk_times((spk_times > bin_ts(n_bins))& ...
            (spk_times < bin_ts(n_bins+1)))))  std(diff(spk_times((spk_times > bin_ts(n_bins))& ...
            (spk_times < bin_ts(n_bins+1)))))];
    end
    plot(bin_ts(1:end-1)*dt,m_std(:,1)*dt);
    xlabel('time (ms)');ylabel('IPT (ms)');
end

end
 


    