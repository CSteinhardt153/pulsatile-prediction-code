function [fr,fr_P,fr_S] = param_to_fr(pulse_rate,S,t_PbP,t_SbP,t_PbS, prt_interact_frac,perc_P_AP,perc_extend)
%%%Parameters to prediction of firing rate fr, fr_P, fr_S
%Should work for any t_SbP, t_PbS, t_PbP, prt_interact_frac (of ipi),
%perc_extend of t_PbP
% pulse rate/pulse amplitude/ spont rate --> fr, fr_S, and fr_P

%Last Updated 1/5/21
%Started 1/5/21
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_steps = 1;
ipi = 1e3./pulse_rate;
%Complete pulse-self block --> induced fr:
corr_pr = pulse_rate./(ceil(1e-3*t_PbP./(1./pulse_rate)));
corr_pr(pulse_rate == 0) = 0;

num_extensions = (floor(t_PbP./(1e3./pulse_rate)));
t_frac =ceil(t_PbP./(1e3./pulse_rate));
t_frac_extend = ceil(((1+ perc_extend.*num_extensions ).^num_extensions).*(t_PbP./(1e3./pulse_rate)));
t_frac_extend((t_frac_extend - t_frac) > 1) = 300.*t_frac_extend((t_frac_extend - t_frac) > 1);
figure(5);plot(t_frac); hold on; plot(t_frac_extend)

%t_PbP being extended should apply to t_PP and t_PS.. and affect partial
%elimination distance and collisions w/ spontaneous -1/5/21

%figure(30); plot(ceil(1e-3*t_PbP./(1./pulse_rate))); hold on;
%plot(ceil(1e-3*t_PbP_tmp./(1./pulse_rate)));

%NEED TO CHANGE TO DISTANCE
part_elim = 1 - min(1,(1 - mod(1e-3.*t_PbP./(1./pulse_rate),1))./prt_interact_frac);
part_elim(mod(1e-3.*t_PbP,(1./pulse_rate)) == 0) = part_elim(mod(1e-3.*t_PbP,(1./pulse_rate)) == 0) + 1;
elim_mult = ceil(1e-3.*t_PbP./(1./pulse_rate))+ part_elim;


if plot_steps
    figure(10);
    plot(pulse_rate,corr_pr,'.--');
    xlabel('PR (pps)'); ylabel('FR (sps)'); title('Pulse self-block')
    
    figure(11); subplot(2,1,1);
    plot(pulse_rate,pulse_rate./ceil(t_PbP./(1e3./pulse_rate))); hold on;
    t_PbP = t_frac_extend.*t_PbP;
    subplot(2,1,2);
    plot(pulse_rate,ceil(1e-3*t_PbP./(1./pulse_rate)))
 %   plot(pulse_rate, ceil(1e-3*t_PbP./(1./pulse_rate))+ part_elim);
end

corr_pAPs = pulse_rate./elim_mult;

%PbS/SbP calculations:
avg_ISI = 1e3/S;
corr_ipis = round(1e3./pulse_rate);
corr_ipis(pulse_rate == 0) = 1;
mod_ms_value = lcm(avg_ISI,max(1,corr_ipis));

%P can block S after any pulse
prob_collide_P_S = min((floor(mod_ms_value./ipi).*t_PbS)./mod_ms_value, 1); %num of p in window on average
%S only can block out of Ps that became APs
%prob_collide_S_P = min((floor(mod_ms_value./corr_ipis).*t_SbP)./mod_ms_value,1);
prob_collide_S_P = min((floor(mod_ms_value./corr_pAPs).*t_SbP)./mod_ms_value,1);

if plot_steps
    figure(12);
    subplot(2,1,1); plot(pulse_rate, prob_collide_P_S);
    subplot(2,1,2); plot(pulse_rate, prob_collide_S_P);
end

S_aftr_blck_by_P = min(S,pulse_rate).*(prob_collide_P_S);%All P could block S of t_PbS high enough
P_aftr_blck_by_S = min(S,perc_P_AP.*corr_pAPs).*(prob_collide_S_P);

fr = max(S + perc_P_AP.*corr_pAPs - S_aftr_blck_by_P - P_aftr_blck_by_S,0);
fr_P = perc_P_AP.*corr_pAPs - P_aftr_blck_by_S;
fr_S = S - S_aftr_blck_by_P ;
end
