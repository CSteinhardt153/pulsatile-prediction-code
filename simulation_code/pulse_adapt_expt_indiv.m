function [spiking_info,fr, avg_CV, avg_ISI] = pulse_adapt_expt_indiv(sim_info,curr_options, pulse_rate, output,change_params, tot_reps,do_parallel,expt,gNs)
%Run individual block of pulsatile stimulation with any number of pulse
%rates and curr_options and see firing rate in each case.
%Can run parellel or not
%
%Started 10/10/20
%Last Update - 10/10/20
%CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l_pulse= length(pulse_rate);
l_cur = length(curr_options);
if do_parallel
    parfor n_currents = 1:l_cur
        n_currents
        for n_pulses = 1:l_pulse %parfor
            change_params =struct();
            change_params.neuron.gNa = gNs(1);
            change_params.neuron.gKL = gNs(2);
            change_params.neuron.gKH = gNs(3);
            %%% for rep_num = 1:tot_reps %just for DC experiment to be parallle
            
            %Flexible place to create a stimulation paradigm to perform on
            %neuron
            
            disp(['Curr:' num2str(curr_options(n_currents)) ' Pulses: ' num2str(pulse_rate(n_pulses)) ])
            
            change_params.Inj_cur = curr_options(n_currents);
            change_params.pulse_rate = pulse_rate(n_pulses);
            
            
            % For do_CV_plot or pulse_adapt case
            for rep_num = 1:tot_reps
                [spiking_info] = run_expt_on_axon_10_10_20_f(sim_info,output,change_params,expt);%single_node_KH_adapt_v1_f(sim_info,output,change_params);
                
                %Check if spiking is from pulses or from spont spikes:
                avg_ISI(n_currents,n_pulses,rep_num) = mean(spiking_info.end.ISI);
                avg_CV(n_currents,n_pulses,rep_num) = spiking_info.end.CV;
                fr(n_currents,n_pulses,rep_num) = spiking_info.fr;
            end
        end
    end
    spiking_info = [];
%     save(sprintf('fr_reps_%s_steps_%s_reg_%s_spking_%s',num2str(tot_reps),...
%         date, num2str(sim_info.is_reg), num2str(sim_info.inj_cur(2))),'fr','avg_CV','avg_ISI','sim_info','curr_options','pulse_rate')
%     
else
    % Do so that you can see output visuals
    for n_currents = 1:l_cur
        for n_pulses = 1:l_pulse
            
            %%% for rep_num = 1:tot_reps %just for DC experiment to be parallle
            
            %Flexible place to create a stimulation paradigm to perform on
            %neuron
            
            disp(['Curr:' num2str(curr_options(n_currents)) ' Pulses: ' num2str(pulse_rate(n_pulses)) ])
            
            change_params.Inj_cur = curr_options(n_currents);
            change_params.pulse_rate = pulse_rate(n_pulses);
            
            % For do_CV_plot or pulse_adapt case
            for rep_num = 1:tot_reps
                % rng(10) %same EPSCS
                [spiking_info] = run_expt_on_axon_10_10_20_f(sim_info,output,change_params);
                if ~isnan(max(spiking_info.end.spk_times))
                spk_time_exact = zeros(1,round(max(spiking_info.end.spk_times)));
                spk_time_exact(round(spiking_info.end.spk_times)) = 1;
                end
                %                    figure(100);
                %                    subplot(6,1,n_pulses)
                %                    plot(spk_time_exact); hold on;
                %                    plot( movmean(spk_time_exact,10));hold on;
                %                    title(sprintf('%s mA %s sps',num2str(pulse_rate(n_pulses)),num2str(spiking_info.fr)));
                %
                % Want all this information too:
                %                     if isempty(expt.num)
                %                         %Observing induced fr/ISI
                %                         figure(20); histogram(spiking_info.end.ISI,10); hold on;
                %                         title(sprintf('mu = %s ms', num2str(sim_info.mu_IPT)));
                %                         xlabel('ISI (ms)');ylabel('Num Spikes');
                %                         set(gca,'fontsize',16)
                %                     end
                %Check if spiking is from pulses or from spont spikes:
                
                avg_ISI(n_currents,n_pulses,rep_num) = mean(spiking_info.end.ISI);
                avg_CV(n_currents,n_pulses,rep_num) = spiking_info.end.CV;
                fr(n_currents,n_pulses,rep_num) = spiking_info.fr;
                
                
                
                %%%%% FOR PLOT OUTS:
                save_img = 0;
                if save_img
                    
                    saveas(gcf, sprintf('I%s_PR%s_V_phase.eps',num2str(-20*curr_options(n_currents)),...
                        num2str(pulse_rate(n_pulses))) , 'epsc');
                end
                
                % Add experiment for finding phase locked pulses:
                check_phase_locking = 0;
                if check_phase_locking
                    clear p_times pulse_timing pl_APs pl_timing fin_pts fin_sts
                    spk_timing = spiking_info.end.spk_times;
                    pulse_timing = spiking_info.end.pulse_time;
                    
                    p_times = pulse_timing(find(diff(pulse_timing) > .5));
                    ipi = min(diff(p_times));
                    %for n_pulses
                    fin_pts(round(p_times*1e3)) = 1;
                    fin_sts(round(spk_timing*1e3)) = 1;
                    
                    %         figure(1);subplot(2,1,2);
                    %         plot(fin_sts,'k'); hold on;title('spike timing');
                    %         subplot(2,1,1);plot(fin_pts,'r'); title(sprintf('pulse timing (IPI = %s)',num2str(ipi)));
                    
                    for n_ps = 1:length(p_times)
                        t_dist_wind = ipi;
                        t_dif_dist = (spk_timing - p_times(n_ps)); %ms
                        pl_APs(n_ps) = sum((t_dif_dist >= 0) & (t_dif_dist <= ipi));
                        if (sum((t_dif_dist >= -t_dist_wind) & (t_dif_dist <= t_dist_wind)) < 1)
                            pl_timing.APs(n_ps).spk = nan;
                        else
                            pl_timing.APs(n_ps).spk = t_dif_dist((t_dif_dist >= -t_dist_wind) & (t_dif_dist <= t_dist_wind));
                        end
                    end
                    
                    pl_timing.nAPs = pl_APs;
                    pl_timing.t_tots = [pl_timing.APs.spk];
                    
                    %         figure(2);
                    %         subplot(3,1,1);
                    %         hist(pl_timing.t_tots); ylabel('N_{pulses}'); xlabel('\Delta t (ms)');
                    %         title(sprintf('Dist from Ps (IPI = %s) ',num2str(ipi)))
                    %         set(gca,'fontsize',14); box off;
                    %         subplot(3,1,2);
                    %         [counts dists] = hist(pl_timing.t_tots,100)
                    %         bar(dists,counts/length(p_times)); ylabel('P_{pulses}'); xlabel('\Delta t (ms)');
                    %         title(sprintf('Dist from Ps (IPI = %s) ',num2str(ipi)))
                    %         xlim([-ipi ipi])
                    %         set(gca,'fontsize',14); box off;
                    %         subplot(3,1,3);
                    %         [counts2 dists2] = hist(pl_APs);
                    %         bar(dists2,counts2/length(p_times));
                    %         title('Num');
                    %         xlabel('Num APs within IPI after a pulse (ms)');
                    %         ylabel('% of pulses with that many')
                    %         xlim([0 ipi])
                    %         set(gca,'fontsize',14); box off;
                    %Save out figures and datA:
                    cd('/Users/cynthiasteinhardt/Dropbox/single-neuron-stim-model/vestibular-neuron-models/vest_model_pulsatile/simpler_format/useful_sim_data/phase_lock_AP_test')
                    
% %                     save(sprintf('pr_pl_pAP_timing_I%s_pr%s_fr%s_mu%s_%s_to_ipi_5_4_21.mat',num2str(curr_options(n_currents)*-20),num2str(round(1e3./ipi)),num2str(spiking_info.fr),...
% %                         num2str(sim_info.mu_IPT),  date),'sim_info','spiking_info','ipi','pl_timing','p_times','spk_timing')
% %                     
                end
    
            end
        end
    end
end
end

