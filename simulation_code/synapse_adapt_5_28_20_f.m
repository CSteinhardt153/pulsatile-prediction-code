%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code based on modeling equations from "Hair-Cell Versus Afferent Adaptation
%in the Semicircular Canals"
%Gets the approximately right adaptation rate and gain!
%R. D. Rabbitt, R. Boyle, G. R. Holstein, and S. M. Highstein
%Turned into function - takes the stimulation current and models adaptation
%timing of it - going to use to change mu and K of EPSC timing as if
%synpase adapts EPSC production
%Last Updated 2/7/20
%Started 2/6/20 CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do not have proper #s for gains (steady state and typical) - looks about
% right - gain  45/20 2.25, 10/20 - .5 so tried to get right ratio around 0
%m_bar - mean interspike interval = mu with frequency f_bar
function [n] = synapse_adapt_5_28_20_f(s,fig_num,sim_info,do_orig)
%sense_scale,
sampling_rate = 1;%ms20;%ms - originally in ms tuned it so seems right
dt = 1e3; % 1 ms not s from 1e-3 ms (timing units of simulation)
%make s proporational to -I_st in the units of (uA)
sim_length = length(s);

%%%% State variables for adaptation
n1(1) = 0; %slow adapting
n2(1)= 0; %fast adapting
n(1) = 0; %combined state variable

%%%%- time constant
tau_1 = sim_info.tau_1*dt;%3.21*dt;%%%BEST1.5*dt;%3*dt; %s --< ms (in Hair cell paper it's 13) this value is from Manca/Aplin
tau_2 = sim_info.tau_2*dt;%0.304*dt;%%%BEST0.53*dt;%s

if do_orig
    %g_o_k %instantaneous gain (Adapted part)
    g_o_1 = .4;% related to negative stimulation adaptation gain
    g_o_2 = .7;%related to fast gain
    g_inf_1 = .01;
    g_inf_2 = .01;
    
    sampling = 2:(sampling_rate*dt):length(s);
    for cur_t = 2:length(sampling)
        
        ds_dt = s(sampling(cur_t)) -s(sampling(cur_t-1));
        %%%ORIGINAL
        dn1_dt = -(1/tau_1)*n1(cur_t-1) + g_o_1*ds_dt + (g_inf_1/tau_1)*s(sampling(cur_t-1));
        dn2_dt = -(1/tau_2)*n2(cur_t-1) + g_o_2*ds_dt + (g_inf_2/tau_2)*s(sampling(cur_t-1));
        
        n1(cur_t) = n1(cur_t-1) + dn1_dt;
        n2(cur_t) = n2(cur_t-1) + dn2_dt;
        
        n(cur_t) = n1(cur_t-1)+n2(cur_t-1)*r(n2(cur_t-1),do_orig);
    end
    
else
    
    %g_o_k %instantaneous gain (Adapted part)
    g_o_1 = sim_info.g_o_1;%related to negative stimulation adaptation gain
    g_o_2 = sim_info.g_o_2;%related to fast gain
    sense_scale = sim_info.sense_scale;
    
     sampling = 1:(sampling_rate*dt):length(s);
%     tmp_s = s(sampling);
%     figure(300); plot(diff(tmp_s)); hold on;
    for cur_t = 2:length(sampling)
        
        ds_dt = s(sampling(cur_t)) -s(sampling(cur_t-1));
        
        dn1_dt = -(1/tau_1)*n1(cur_t-1) ...%+ g_o_1*ds_dt ...
            + g_o_1*isChange(ds_dt)*(ds_dt - sense_scale*s(sampling(cur_t-1)));%+ ...;
     %  dn2_dt(cur_t) = -(1/tau_2)*n2(cur_t-1) + g_o_2*ds_dt;
        
        %Past sensitization ideas:(also falls out of low amplitude
        %stimulation)
         dn2_dt(cur_t) =-(1/tau_2)*n2(cur_t-1)+g_o_2*ds_dt;%*r(ds_dt,do_orig);
         %-(1/tau_2)*n2(cur_t-1) + g_o_2*ds_dt ...
      % +g_o_2*isChange(ds_dt)*sense_scale*( ds_dt - s(sampling(cur_t-1)))*r(ds_dt,do_orig);
         %+ g_o_2*ds_dt*(sense_scale*( ds_dt -s(sampling(cur_t-1))))*r(ds_dt,do_orig);
       %  -(1/tau_2)*n2(cur_t-1) + g_o_2*isChange(ds_dt)*( ds_dt - s(sampling(cur_t-1))*r2(ds_dt)); %     3/18/20 best model
        % g_o_2*ds_dt*(sense_scale*( ds_dt -s(sampling(cur_t-1))))*r(ds_dt,do_orig);
        %  g_o_2*(1 + sense_scale*(-s(sampling(cur_t-1)) + ds_dt))*ds_dt;% ...
        %  sensitization by subtraction
        %   g_o_2*(1- sense_scale*s(sampling(cur_t-1))*ds_dt)*ds_dt;% ...
        %   sensitization with scaling
        n1(cur_t) = n1(cur_t-1) + dn1_dt;
        n2(cur_t) = n2(cur_t-1) + dn2_dt(cur_t);
        n(cur_t) = n1(cur_t-1)+n2(cur_t-1)*r(n2(cur_t),do_orig);%+g_tot*s(sampling(cur_t));%
        % end
    end
end
% %
% % % % % % % Plot to check decay is working
%  real_time  = [1:(sampling_rate*dt):length(s)]/dt;%/sampling_rate;
%  figure;
%  subplot(3,1,1);
% % plot(real_time,s(1:(sampling_rate*dt):length(s)),'linewidth',2); hold on;
%  plot(real_time,[0 (s(sampling(2:end)) -s(sampling(1:end-1)))]); hold on;%,'%r','linewidth',2)
% %
%  subplot(3,1,2);
%  plot(real_time,n,'linewidth',2); xlabel('time(ms)'); hold on;
% % ylim([-10 10])
%  subplot(3,1,3);
%   plot(n2,'r'); hold on; plot(n1,'b')
% % ylim([-10 10])

settings.g_o_1 = g_o_1;
settings.g_o_2 = g_o_2;

end

function [non_lin_scale] = r(x,do_orig)
%nonlinear excit/inhib asymmetry in fast adaptation
if x >= 0
    non_lin_scale = 1 ;
else
    if do_orig
        non_lin_scale = 0.;
    else
        non_lin_scale = .1;%.1;
    end
    %%%%%0.1; %all previous cases - inhibitory dominated by slow time constant
end
end
% 
%  function [non_lin_scale] = r2(ds_dt_c)
%  %nonlinear excit/inhib asymmetry in fast adaptation
% if ds_dt_c >= 0
%     non_lin_scale = .8;
% else
%     non_lin_scale = 0;%no inhibitory sensitivity
% end
%  end

function [has_change] = isChange(ds_dt_cur)
%nonlinear excit/inhib asymmetry in fast adaptation
if ds_dt_cur ~= 0
    has_change = 1 ;
else
    has_change = 0; % inhibitory dominated by slow time constant
end
end
