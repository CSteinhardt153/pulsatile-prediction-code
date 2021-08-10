function [spiking_info] = run_expt_on_axon_10_10_20_f(sim_info,output,change_params,expt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function version of the KH modle in an axon from "A biophysical model examining the role of
%low-voltage-activated potassium currents in shaping the responses of
%vestibular ganglion neurons -Ariel E. Hight and Radha Kalluri
%Hight & Kalluri is an adaptation of "The Roles Potassium Currents Play in
%Regulating the Electrical Activity of Ventral Cochlear Nucleus Neurons"
% Jason S. Rothman and Paul B. Manis :
%https://www.physiology.org/doi/pdf/10.1152/jn.00127.2002
%This code turn single-compartment model into axon
%Many of the equations required to make the model work are IN R & M
%Took semi-working single node from 10/27 and used to make model in axon
%propagating.
%There were a number of typos and missing pieces of information in the
%original K+H model, including missing variables and negative equations.
%Geometry equations come from Frijns et al (1994), and the d,l, L terms
%come from vestibular models that give specs for each and Goldberg papers (
%in notes ~(11/7/19)
%Turning into functional form with which can run experiments on reaction ot
%current steps on regular, etc.
%%%11/26 - step back from full length axon model to single node model to
%%%attempt to find version that works as single node (as in KH)
%Adding in adaption of the mu of firing using synapse_adapt_f.m
%Same as 5/28/20 except copied for pulsatile work
%Started 2/7/20
%Last Updated 7/19/20 - CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Simulation setup
dt = 0.001;%ms - time step of whole simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Below for spontaneous firing
%%%%%%%%%%%%%%%%%%%%%%Resting conditions:
%rng(10);
[axon_inputs change_params] = make_axon_inputs_7_19_20(change_params, sim_info);    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cm = 0.9; % uF/cm2 - capacitance per SA
S = 10e-6/Cm;%pi*d*l; % becuase 10 pF total

g_Na_bar = change_params.neuron.gNa;
g_KL_bar = change_params.neuron.gKL;
g_KH_bar = change_params.neuron.gKH;
%Current not adding other types of K channels except KH
g_leak = .03; %1e-4;%.03;
%Reversal Potentials
E_k = -81;% mV %-70
E_Na = 82; % mV % 55
E_leak = -65; % mV %-65

Gamma = 0.5; %L s upsidedown L)
V_rest = -65;%mv (from fights in H&K)
%KL
w(1) =  (1+exp(-(V_rest+44)./8.4)).^(-1/4);
z(1) = (1-Gamma).*(1+exp((V_rest+71)/10)).^-1 + Gamma;
%KH
n(1) = (1+exp(-(V_rest+15)/5))^-(0.5);
p(1) =  (1+exp(-(V_rest+23)/6))^-1;
%Na
m(1) = (1+exp(-(V_rest+38)/7))^-1;
h(1) = (1+exp((V_rest+65)/6))^-1;
%h
r(1) = (1 + exp(-(V_rest + 100)/7))^-1;

V(1) = V_rest;

for t = 1:round(sim_info.sim_time/dt) %time looping
        
    %%%%%% AXONAL CURRENTS
    I_leak(t) = g_leak*S*(V(t)-E_leak); %%leak current mV*mS-->(uA)
    %I_KL,I_KH, I_Na, I_epsc
    %I_epsc -synaptic;I_Na %transient sodium current; I_KL %low voltage-gated potassium current; I_KH %high voltage gated potassium current
    %I_i = g_i*V*S*(V-E_i); %for all voltage-dependent currents
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%I_KL: Low Voltage Gated K+ Current (Kv1)
    %g_KL = g_KL_bar * w^4*z; %Kv1: w-act, z-inact; (mS/cm2)
    I_KL(t) = g_KL_bar*S*w(t).^4.*z(t).*(V(t)-E_k);
    %%%%Kv1 - activate quickly, inactivate half max value slow time
    %x_inf represents some alpha/(alpha+beta).tau_x = 1/(alpha + beta)
    w_inf = (1+exp(-(V(t)+44)./8.4)).^(-1/4);
    z_inf = (1-Gamma)*(1+exp((V(t)+71)./10)).^-1 + Gamma;
    tau_w = 100*(6*exp((V(t)+60)./6) + 16*exp(-(V(t)+60)./45)).^(-1) + 1.5;
    tau_z = 1000*(exp((V(t)+60)./20)+ 16*exp(-(V(t)+60)./8)).^(-1) + 50;
    % NEED THESE FOR THE NULLCLINES
    dw_dt = (w_inf - w(t))./tau_w; %Modifies for Kv7
    dz_dt = (z_inf - z(t))./tau_z; %Modifies for KLA
    %%%%Kv7 - slower activation of Kv7 (KCNQ) currents (rest same)
    %H&K claims that adding in Kv7 had subtle difference on conduction
    %especially at measures 8:2 ratio so for now I_KL is only Kv1
    %tau_w_kv7 = 1000*(6*exp((V(t)+60)/6) + 16*exp(-(V(t)+60)/45))^-1 + 1;
    %z_inf_kv7 = 0; %no inactivation
    %%%%KLA  - Rapidly inactivating
    %Also supposedly little difference but raises threshold for AP
    %Gamma = 0; % with Tau-z_A gives I_KLA properties
    %tau_z_A = 10*(exp((V(t)+60)/20) + 16*exp(-(V(t)+60)/8))^-1 + 50;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%I_KH: High Voltage Gated K+ Current
    %Kv1:Kv7 8:2 ratio
    phi = 0.85; %g_KL = g_KL_bar*w^4*z;
    I_KH(t) = g_KH_bar*S*(phi.*n(t).^2 + (1-phi).*p(:,t)).*(V(:,t)-E_k);
    n_inf = (1+exp(-(V(t)+15)./5)).^-(0.5);
    p_inf = (1+exp(-(V(t)+23)./6)).^-1;
    tau_n = 100*(11*exp((V(t)+60)./24) + 21*exp(-(V(t)+60)./23)).^(-1) + 0.7;
    % tau_p = 100*(4*exp((V(t)+60)./32) + 16*exp(-(V(t)+60)./22)).^(-1) + 50;%- KH original
    
    %%% USING THIS:
    tau_p = 100*(4*exp((V(t)+60)./32) + 5*exp(-(V(t)+60)./22)).^(-1) + 5; %- RM original
    
    % tau_p = 100*(4*exp((V(t)+60)./32) + 7*exp(-(V(t)+60)./22)).^(-1) + 10;
    %  %5  -my adjusted RM/KH intermediary that matched Fig 4/5
    dn_dt = (n_inf - n(t))./tau_n; %EQ form dx_dxt = (x_inf - x)/tau_x from R&M
    dp_dt = (p_inf - p(t))./tau_p;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%I_Na: Fast Na+ Current (HH):
    I_Na(t) = g_Na_bar.*m(t).^3.*h(t)*S.*(V(t)-E_Na);
    m_inf = (1+exp(-(V(t)+38)./7)).^-1;
    h_inf = (1+exp((V(t)+65)./6)).^-1; %error in text
    tau_m = 10*(5*exp((V(t)+60)./18) + 36*exp(-(V(t)+60)./25)).^(-1) + 0.04;
    tau_h = 100*(7*exp((V(t)+60)./11)+10*exp(-(V(t)+60)./25)).^(-1) + 0.6;
    dm_dt = (m_inf - m(t))./tau_m;
    dh_dt = (h_inf - h(t))./tau_h;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if sim_info.doIh
        E_h = -42;%mV
        g_h_bar = .1;%0.1;%mS/cm2
        I_h(t) = g_h_bar.*r(t).^3.*S.*(V(t)-E_h);
        r_inf = (1 + exp(-(V(t) + 100)/7))^-1;
        tau_r = 10^5 * ((237*exp((V(t)+60)/12)) + 17*exp(-(V(t)+60)/14)^-1) + 25;
        dr_dt = (r_inf - r(t))./tau_r;
                %Injected current steps I_inj), Synaptic current (I_epsc)
                %with I_h
        dV_dt =  (1/(Cm*S))*(-I_KL(t)-I_Na(t)-I_KH(t)-I_leak(t) - I_h(t));%); %
    else
    %Injected current steps I_inj), Synaptic current (I_epsc)
    dV_dt =  (1/(Cm*S))*(-I_KL(t)-I_Na(t)-I_KH(t)-I_leak(t));%); %--> V
    end
    
    if sim_info.inj_cur(1)  %External current spread across membrane
        
        dV_dt = dV_dt + (1/(Cm*S))*axon_inputs.V_stim(t);%(mA)
    end
    if sim_info.inj_cur(2) %EPSCs ( jsut at synapse)
        dV_dt = dV_dt+(1/(Cm*S))*axon_inputs.I_epsc(t); %(uA)--> V - in KH from Eq 1
    end
    %  dV_dt(synapse_loc+sim_info.pos_shift) = dV_dt(synapse_loc+sim_info.pos_shift)+sim_info.syn_shift;
    %Can also add I_inj into here!
    V(t+1) = V(t) + dt*dV_dt;%V/s*ms --> mV
    
    %Updating Gating variables/Voltage
    w(t+1) = w(t) + dt*dw_dt; %%KL
    z(t+1) = z(t) + dt*dz_dt;
    n(t+1) = n(t) + dt*dn_dt; %%KH
    p(t+1) = p(t) + dt*dp_dt;
    m(t+1) = m(t) + dt*dm_dt; %%Na
    h(t+1) = h(t) + dt*dh_dt;
    
     if sim_info.doIh
    r(t+1) = r(t) + dt*dr_dt; %%h
    end
end


%% Visualizing specific nodes/ propagation

%view_node = synapse_node;
%end_elec = synapse_loc+7;%size(V,1)-1;
%view_node =  sim_info.stim_loc_choice;%synapse_loc+1; %end_elec;

%Confirm if AP propagates
[is_spike,spk_hght]  = detect_spikes(V, dt);

if ~sim_info.isDC
    spk_ts = (find(is_spike));
    %blank out around artifact zone so that
    for n_pulse = 1:length(change_params.pulse_timing)
        
        spk_ts(ismember(spk_ts,[change_params.pulse_timing(n_pulse)-300:change_params.pulse_timing(n_pulse)+400])) = [];
        %not_spks = find((spk_ts > pulse_timing(n_pulse)-10) & (spk_ts < pulse_timing(n_pulse)+300));
        %spk_ts(spk_ts not_spks) = [];
        
    end
    
    is_spike = zeros(1,sim_info.sim_time/dt);
    is_spike(spk_ts) = 1;
end

coeff_var = @(v_std,v_mean) v_std./v_mean;
% figure(12); plot(V(end_elec,:)); hold on;
% plot(find(is_spike),V(end_elec,find(is_spike)),'.');
% figure(13);  plot(V(cond_strt_node+1,:)); hold on;
% plot(find(is_sp),V(cond_strt_node+1,find(is_sp)),'.');
%Spike at end of synapse
if isempty(find(is_spike))
    avg_ISI = nan;
    coef_var_ISI = nan;
    fin_spk_times = nan;
else
    
    fin_spk_times = find(is_spike)*dt;
    fin_spk_times(find(diff(fin_spk_times) <.3)+1) = [];
    
    %CHANGE:
     %replace:fin_spk_times = fin_spk_times(fin_spk_times > 30);%sim_info.sim_start_time);
    %Only count spiking in the range:
    if ~output.all_spk
    if isempty(sim_info.sim_end_time)
       fin_spk_times = fin_spk_times(fin_spk_times > sim_info.sim_start_time);
    else
        fin_spk_times = fin_spk_times((fin_spk_times > sim_info.sim_start_time) & ...
            (fin_spk_times < sim_info.sim_end_time));
    end
    end
    %%%ISI (interspike interval)
    ISI = diff(fin_spk_times);
    avg_ISI = mean(ISI);
    coef_var_ISI = coeff_var(std(ISI),mean(ISI));
end

num_spk = sum(~isnan(fin_spk_times));
if isempty(sim_info.sim_end_time)
    fr = num_spk/((sim_info.sim_time-sim_info.sim_start_time)*dt);
else
    fr = num_spk/((sim_info.sim_end_time - sim_info.sim_start_time)*dt);
end
%Output values
spiking_info.end.pulse_time = find(axon_inputs.V_stim > 0)*dt;
spiking_info.end.spk_times = fin_spk_times;
spiking_info.end.ISI = diff(fin_spk_times);
spiking_info.end.CV = coef_var_ISI;
spiking_info.fr = fr;
%spiking_info.V = V;

%disp(sprintf('CV %s',num2str(coef_var_ISI)))
disp(sprintf('ISI %s, CV %s fr:%s',...
    num2str(avg_ISI),num2str(coef_var_ISI),num2str(fr)))


%Plot simulation in simulation timing ( look at V etc.)
   show_rel_timing = 0;
if output.vis_plots
    sim_timing = [0:dt:sim_info.sim_time];
        if ~sim_info.inj_cur(1) & sim_info.inj_cur(2)
            I_inj_tot = 1e-6*axon_inputs.I_epsc/(Cm*S); %pA
            is_EPSC = 'has';
            ylabel('Total I_{inj} [at axon] (pA)')
        else
        if sim_info.inj_cur(1) & ~sim_info.inj_cur(2)
            
            I_inj_tot = axon_inputs.V_stim; % I_stim(mA)
            is_EPSC = 'no';
        end
        if sim_info.inj_cur(1) & sim_info.inj_cur(2)
            I_inj_tot =  axon_inputs.V_stim+axon_inputs.I_epsc; % (mA+mA)
            is_EPSC = 'has';
        end   
    
    
    
   %(output.vis_plot_num)
        
        just_ps = 0;
         if just_ps
              figure;
         subplot(2,1,1);
          plot(sim_timing(1:end-1),I_inj_tot*1e6);  hold on;
            xlim([100 400]); box off; %    xlim([150 400]); box off;
        ylabel('Total I_{inj} [at axon] (mA)')
          subplot(2,1,2);
          plot(sim_timing(1:length(V)), V); hold on;%'k'); hold on;
        if ~isnan(fin_spk_times)
            plot(fin_spk_times,V(round(fin_spk_times/dt)),'.');
        end
        xlabel('time (ms)'); ylabel('V (mV)');
         box off;
          xlim([100 1000]);box off;
         end
        
          figure;
     if show_rel_timing
        ax1 = subplot(4,1,1); 
     else
        ax1 = subplot(6,1,1);
     end
        
            plot(sim_timing(1:end-1),I_inj_tot*1e6);  hold on;
            xlim([100 400]); box off; %    xlim([150 400]); box off;
        ylabel('Total I_{inj} [at axon] (mA)')
        end
        which_s = 1;
        %title(sprintf('%s EPSC %d,I= %s,PR = %s ',is_EPSC,which_s,num2str(change_params.Inj_cur),num2str(change_params.pulse_rate)))
        title(sprintf('%s EPSC %d,I= %s',is_EPSC,which_s,num2str(change_params.Inj_cur)))
      
        if   show_rel_timing
        ax2 = subplot(4,1,2:3);
        else
        ax2= subplot(6,1,2:3);
        end
        %subplot(3,1,2);subplot(3,1,2:3)%
        % plot(sim_timing(1:size(V,2)), V(output.view_node_1,:)); hold on;
        plot(sim_timing(1:length(V)), V); hold on;%'k'); hold on;
        if ~isnan(fin_spk_times)
            plot(fin_spk_times,V(round(fin_spk_times/dt)),'.');
        end
        xlabel('time (ms)'); ylabel('V (mV)');
         box off;
          xlim([100 400]);box off;
      % Show mod results for regular/irregular:
 
     if show_rel_timing
         ipis = unique(diff(spiking_info.end.pulse_time));
         ipi= mean(ipis(ipis > .002));
         pr = 1e3/ipi;
         rel_t_spk = mod(spiking_info.end.spk_times,ipi);
         subplot(4,1,4); histogram(rel_t_spk,15)
         box off;
         
         save_dir = '/Users/cynthiasteinhardt/Dropbox/Fridman_lab/submissions/pulsatile/pp_sp_ps_paper/';
         
      %%%   print(gcf, fullfile(save_dir, sprintf('I%s_P%s_reg%s_5_10_21.pdf',num2str(change_params.Inj_cur*-20),...
       %%%         num2str(change_params.pulse_rate),num2str(sim_info.is_reg))), '-dpdf', '-r0');
         %   saveas(gcf, fullfile(save_dir, sprintf('I%s_P%s_reg%s_5_10_21.eps',num2str(change_params.Inj_cur*-20),...
         %       num2str(change_params.pulse_rate),num2str(sim_info.is_reg))), 'epsc');

     else
         
      %   SHOW THE DYNAMICS:
       ylim([-100 100])
       ax3 =subplot(6,1,4); %subplot(3,3,7);
        plot(sim_timing,w); hold on;
        plot(sim_timing,z);
        title('I_{KL}')
     %       xlim([150 400])
            ylim([0 1])
            legend('w','z')
        ax4 = subplot(6,1,5);%subplot(3,3,8);
        plot(sim_timing,n,'k');hold on;
        plot(sim_timing,p);
        title('I_{KH}')
     %       xlim([150 400])
                  ylim([0 1])
                  legend('n','p')
        ax5 = subplot(6,1,6);%subplot(3,3,9);
        plot(sim_timing,m);hold on;
        plot(sim_timing,h);
        title('I_{Na}')
     %   xlim([150 400])
              ylim([0 1])
        linkaxes([ax1 ax2 ax3 ax4 ax5],'x')
        legend('m','h')
     end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Output specification (phase plane)
%% Phase plane analysis:
if output.do_phase_plane_anal

    figure(output.pp_plot); subplot(3,2,1);
    vis_seq = round(linspace(1,length(V),5000));
    cols_tot = jet(length(vis_seq));
    plot3(1e-3*vis_seq ,V(vis_seq),n(vis_seq)); hold on;
    xlabel('t (ms)'); ylabel('n'); zlabel('V'); title(sprintf('KH, gKH = %s',num2str(change_params.neuron.gNa)));
    box off;
    subplot(3,2,2);
    plot3(1e-3*vis_seq ,V(vis_seq),p(vis_seq)); hold on;
    xlabel('t (ms)'); ylabel('p'); zlabel('V');
    %plot(V,p);hold on;
    %ylabel('p'); xlabel('V');
    title('KH');
    box off;
    subplot(3,2,3);
    plot3(1e-3*vis_seq ,V(vis_seq),m(vis_seq)); hold on;
    xlabel('t (ms)'); ylabel('m'); zlabel('V');
    title('Na');
    box off;
    subplot(3,2,4);
    plot3(1e-3*vis_seq ,V(vis_seq),h(vis_seq)); hold on;
    %plot(V,h);hold on;
    ylabel('h'); zlabel('V');title('Na');
    box off;
    subplot(3,2,5);
    plot3(1e-3*vis_seq ,V(vis_seq),w(vis_seq)); hold on;
    %plot(V,w);hold on;
    ylabel('w'); zlabel('V'); title('KL');
    box off;
    subplot(3,2,6);
    plot3(1e-3*vis_seq ,V(vis_seq),z(vis_seq)); hold on;
    %plot(V,z);hold on;
    ylabel('z'); zlabel('V');title('KL');
    box off;
    %%%%%%%%% Analysis with I_kl, I_kh, I_na v. V to see how change with irreg and reg:
    
    %     figure(output.pp_plot*10);
    %     subplot(3,1,3);
    %     for n_t = 1:length(vis_seq)
    %         plot(V(vis_seq(n_t)),I_KL(vis_seq(n_t)),'.-','color',cols_tot(n_t,:,:)); hold on;
    %     end
    %    % plot(V(1:end-1),I_KL);  hold on;
    %     xlabel('V');ylabel('I_{KL}');
    %     subplot(3,1,2);
    %     for n_t = 1:length(vis_seq)
    %         plot(V(vis_seq(n_t)),I_KH(vis_seq(n_t)),'.-','color',cols_tot(n_t,:,:)); hold on;
    %     end
    %     %plot(V(1:end-1),I_KH);  hold on;
    %     xlabel('V');ylabel('I_{KH}');
    %     subplot(3,1,1);
    %     for n_t = 1:length(vis_seq)
    %         plot(V(vis_seq(n_t)),I_Na(vis_seq(n_t)),'.-','color',cols_tot(n_t,:,:)); hold on;
    %     end
    %     %plot(V(1:end-1),I_Na); hold on;
    %     xlabel('V');ylabel('I_{Na}');
    %     title(sprintf('I v. V Plots, gKH = %s, gKL = %s',num2str(change_params.neuron.gKH),num2str(change_params.neuron.gKL)));
end

%Label APs as coming from pulses or spontaneous activity
if output.label_aps
    %Find pusle timing and spikes within them.
    %Find spontaneous timing
    %Find spikes within ~2 ms (observe loose cut off)
    pevk_spk_idxs = zeros(size(change_params.pulse_timing));
    for n_pulse = 1:length(change_params.pulse_timing)
        spiking_info.p_times = change_params.pulse_timing;
        pevk_spk = find(((spiking_info.end.spk_times - (change_params.pulse_timing(n_pulse)*1e-3)) < 2) ...
            &  ((spiking_info.end.spk_times - (change_params.pulse_timing(n_pulse)*1e-3)) > 0));
        if ~isempty(pevk_spk)
            pevk_spk_idxs(n_pulse) = pevk_spk;
        else
            pevk_spk_idxs(n_pulse) = nan;
        end
        clear pevk_spk
    end
    spiking_info.pevk_spk = pevk_spk_idxs;
end
%sprintf('Cond in %s ms at %s m/s',num2str(cond_time),num2str(cond_vel))

%% Functions used for above
% %Spike Detection
    function [is_spike,spk_hght] = detect_spikes(V, dt)
        delta_t = 0.01;%ms
        C1 = -20;
        C2 = 20; %mV
        %C1_epsp = -10; C2_epsp = 10; %mV
        last_spk = 1/dt; %ms threshold for last spike distance
        dt_steps = delta_t/dt;
        dt_step2= 1.75/dt;%ms
        n_spk =1;
        min_V = -35;
        spk_hght=  []; is_spike= [];
        for t_o = (1+dt_step2):(length(V) - dt_step2)
            if (V(t_o) > min_V) & (V(t_o) > V(t_o + dt_steps)) & (V(t_o) > V(t_o - dt_steps)) %&(V(t_o) > 0)
                %...& (V(t_o) < 120) ...   & (abs(V(t_o)- V(t_o +1)) < .25)
                is_spike(t_o) = 1;
                % is_EPSP(t_o) = 1;
                
                spk_hght(n_spk,1) = (V(t_o+dt_step2) - V(t_o));
                spk_hght(n_spk,2) = (V(t_o) - V(t_o-dt_step2));
                n_spk = n_spk + 1;
                
                %Is a spike
                if ((V(t_o+dt_step2) - V(t_o)) < C1) &...
                        ((V(t_o) - V(t_o-dt_step2)) > C2)
                    is_spike(t_o) = 1;
                    %  is_EPSP(t_o) = 0;
                else
                    is_spike(t_o) = 0;
                end

            else
                is_spike(t_o) = 0;% is_EPSP(t_o) = 0;
            end
            
        end
        
    end

    
end
