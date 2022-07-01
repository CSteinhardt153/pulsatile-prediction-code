%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ephys_data_analysis_2_28_21.m
%Continuation of analysis in data_analysis_comp_model.m
%%% Analyzing the 3 sets of data from mitchell_data folder for
%%%%%% evidence of bends in pulse amp/rate fr relationship
%%%%%% amplitude effects across neurons
%%%%%% "state" of neuron influencing level of bends

%Now trying 3 neurons with diff amplitud response data. Know the actual
%amplitudes used. 2 neurons are neurons 1 and 2 from previous study.
% 1) Test if code uses similar amplitudes or amplitude ratios
% 2) Test for similar interactions
% 3) Turn analyses into functions

%Started 11/10/20 CRS / Renamed 2/28/21
%Last Updated 8/10/21 CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
base_dir = cd;
data_dir = '/Users/cynthiasteinhardt/Dropbox/single-neuron-stim-model/vestibular-neuron-models/vest_model_pulsatile/simpler_format/mitchell_data_all';
%cd(data_dir);

%addpath('/Users/cynthiasteinhardt/Dropbox/single-neuron-stim-model/vestibular-neuron-models/vest_model_pulsatile/simpler_format');
cur_range= [0 300];
% Data from different current amplitudes (2 & 3)
% Folders:
orig_data_fld = 'afferent_data';
amp_data_fld = {'afferents_data_2','afferents_data_3'};

% Data from afferent_data_updated.xls
%Filename	afferent ID	mean_FR	mean_ISI	std_ISI	CV	CV*	Classification
dat_types = {'mean_fr','mean_isi','std_isi','cv','cv*'};
afferent_info = [32.86,	30.37,	21.80,	0.72,	0.42,
    43.34	23.08	11.02	0.48	0.35
    31.37	31.98	11.01	0.34	0.18
    26.16	38.04	18.93	0.50	0.20
    21.17	46.73	9.33	0.20	0.02
    19.24	51.86	21.41	0.41	0.04
    25.34	39.45	17.88	0.45	0.16
    nan     nan     nan     nan     nan]%ir
%have data from aff8 but not this info from mitchell excel
is_reg = (afferent_info(:,5) < 0.1);

orig_neurs = [1:6];
cur_amp_neurs = [1 2 7 4 6 8];

%%%%% Current Amplitude Info (amp test 1)
curr_perc = [25	50	75	87.5 100];
amps_per_neur = [40	80	120	140	160
    42	84	126	nan	168
    42	84	126	nan	168];


model_name = 'best_sim_fit_trefs_10_27_20.mat';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single amplitude data:
aff_dats = dir(fullfile(fullfile(data_dir,orig_data_fld),'affer*.mat'))
aff_names = {aff_dats(:).name}
name_info  =cellfun(@(x) strsplit(x(9:end),'.'),aff_names,'UniformOutput',false);
aff_nums = unique(cellfun(@(x) str2num(x{1}),name_info));

n_plot = 20;


I_range = 0:1:400;

for n_neurs = 1:length(aff_nums)
    aff_file_names = dir(fullfile(aff_dats(1).folder,sprintf('afferent%s*',num2str(aff_nums(n_neurs)))))
    data_per_aff = load(fullfile(aff_file_names.folder,aff_file_names.name));
    min_isi(n_neurs) = min(diff(find(data_per_aff.ua)))
end

for n_neurs = 1:length(aff_nums)
    aff_file_names = dir(fullfile(aff_dats(1).folder,sprintf('afferent%s*',num2str(aff_nums(n_neurs)))));
    data_per_aff = load(fullfile(aff_file_names.folder,aff_file_names.name));
    plot_it= 0;
    %%[fin_nums base_fr] = data_t_fr(data_per_aff, plot_it,n_neurs,n_neurs,length(aff_nums),afferent_info,n_plot);
    [fin_nums base_fr] = data_split(data_per_aff,plot_it,n_neurs,n_neurs,length(aff_nums),afferent_info,n_plot);

    indiv_slopes(n_neurs) = fin_nums.slope;%m_per_amp;
    clear fr_preds
    S_cur = base_fr(1) + .5*base_fr(2);
    for n_Is = 1:length(I_range)
        I_cur = I_range(n_Is);
           [tot_pred ] = interp_pred_f_5_5_21(I_cur,S_cur,fin_nums.pr(:,1)');
       % [tot_pred] = interp_pred_fr_v2(I_cur,S_cur,fin_nums.pr_n(:,1)');
        %[tot_pred] = interp_pred_fr(I_cur,S_cur,fin_nums.pr_n(:,1)');
        %[tot_pred] = pred_fr_from_pr_I_S(I_cur,S_cur,fin_nums.pr_n(:,1)');
        fr_preds(n_Is,:) = S_cur + tot_pred;
    end
    
    rms_per_I = rms((fr_preds - fin_nums.fr(:,1)')');
    [rms_min idx_min]=min(rms_per_I);
    I_best = I_range(idx_min);
    rms_best = rms_min;
    
    %[full_pred] = interp_pred_fr_v2(I_best,S_cur,[0:300]);
    [full_pred ] = interp_pred_f_5_5_21(I_best,S_cur,[0:300]);
    
    [I_idx,rms_idx,fr_pred_best] = two_d_rms_eval(S_cur,fin_nums.pr(:,1),fin_nums.fr(:,1));
    
    %Previous modeling:
    % [fin_fit best_run] = best_fr_prev_model(model_name,base_fr,fin_nums,n_neurs,1,1,cur_range,0,base_dir);
    disp(sprintf('Aff%d',n_neurs))
    disp(fin_nums.pr(:,1)')
    figure(20); subplot(2,3,n_neurs);
    errorbar([0; fin_nums.pr(:,1)],[base_fr(:,1); fin_nums.fr(:,1)],...
        [base_fr(:,2); fin_nums.fr(:,2)],'k.');
    hold on;
    rms_per_single(n_neurs) = rms_idx;
   % plot([0; fin_nums.pr_n(:,1)],[base_fr(:,1); fin_nums.fr_n(:,1)],'k');
    %    plot(fin_nums.pr_n(:,1),fin_fit.sim_cur(fin_fit.idx).fr,'m')
    %   plot(fin_nums.pr_n(:,1), fin_fit.sim_cur(fin_fit.idx).P,'r--')
    %   plot(fin_nums.pr_n(:,1), fin_fit.sim_cur(fin_fit.idx).S,'b--')
    %    plot([0; fin_nums.pr_n(:,1)],[base_fr(:,1); fr_preds(idx_min,:)'],'r');
    
    %plot([0:300],full_pred+base_fr(:,1),'m--');
    plot([0:400],fr_pred_best,'r--');
    xlabel('Pulse Rate'); ylabel('Firing Rate'); box off;
    title(sprintf('Afferent %s: CV*: %s, I: %s',num2str(aff_nums(n_neurs)), num2str(round(afferent_info(n_neurs,5),2)),num2str(I_best)))
    
    
end
%RMS per neuron: 
rms_per_single


%% Amp dat analyses: build the dataset from the files:

for n_fld = 1:length(amp_data_fld)
    %cd(fullfile(data_dir,amp_data_fld{n_fld}))
    aff_dats= dir(fullfile(data_dir,amp_data_fld{n_fld},'affer*.mat'));
    aff_names = {aff_dats(:).name};
    name_info  =cellfun(@(x) strsplit(x(9:end),'_'),aff_names,'UniformOutput',false);
    aff_nums = unique(cellfun(@(x) str2num(x{1}),name_info));
    
    for cur_neur=1:length(aff_nums)
        aff_file_names = dir(fullfile(data_dir,amp_data_fld{n_fld},sprintf('afferent%s*',num2str(aff_nums(cur_neur)))));
        
        clear cur_neur_amp
        for n_amps = 1:length(aff_file_names) %amps for each afferent
            tmp_curr_string = strsplit(aff_file_names(n_amps).name,'_');
            aff_num = str2num(tmp_curr_string{1}(9:end));
            
            if aff_num == 3
                aff_num = 7; % I think from the excel
            end
            
            neur_idx = find(cur_amp_neurs == aff_num);
            tmp_curr_string = strsplit(tmp_curr_string{2},'u');
            cur_neur_amp(n_amps) = str2num(tmp_curr_string{1});
            %Extract data from file
            data_per_aff = load(fullfile(data_dir,amp_data_fld{n_fld},aff_file_names(n_amps).name));
            if n_fld == 2
                data_per_aff.stim = data_per_aff.stim';
            end
        %    figure(n_fld*100+cur_neur);
        %        subplot(length(aff_file_names),1,n_amps)
        %        disp(['I ' num2str(n_amps)]);
             [fin_nums base_fr] = data_split(data_per_aff,0,neur_idx,aff_num,length(cur_amp_neurs),afferent_info,20);
            
           %[fin_num1 base_fr ] = data_t_fr(data_per_aff,0,neur_idx,aff_num,length(cur_amp_neurs),afferent_info,20);%40);           
           
           
           neuron_data(neur_idx).amp(n_amps).pr = fin_nums.pr;
            neuron_data(neur_idx).amp(n_amps).fr = fin_nums.fr;
            neuron_data(neur_idx).amp(n_amps).base_fr = base_fr;
            neuron_data(neur_idx).amp(n_amps).amp = cur_neur_amp(n_amps);
            neuron_data(neur_idx).amp(n_amps).dfr = fin_nums.dfr;
            neuron_data(neur_idx).amp(n_amps).slope = fin_nums.slope;
%             
          %plot slopes over top:
          plot([0:350],(base_fr(1) + neuron_data(neur_idx).amp(n_amps).slope.*[0:350]),'r')
        end
        % %         %Maybe should be fitted with average firing rate across all trial with same
        % %         %neuron
        % %         for n_amp = 1:length(aff_file_names)
        % %             overall_base_fr =mean(vertcat(neuron_data(neur_idx).amp.base_fr));
        % %             tmp_fin_nums.pr_n = neuron_data(neur_idx).amp(n_amp).pr;
        % %             tmp_fin_nums.fr_n = neuron_data(neur_idx).amp(n_amp).fr;
        % %             %cd(base_dir);
        % %             cur_range= [0 300];
        % %         end
    end
end

%% All slopes:
slopes_all = [];

for n_neur= 1:6
slopes_all = [slopes_all neuron_data(n_neur).amp.slope];
end
slopes_all = [slopes_all  indiv_slopes];
figure(10);histogram(slopes_all,10);
xlim([-.25 1]); box off;
xlabel('Slope with Pulse Rate (sps/pps)'); ylabel('Num Recordings')
set(gca,'fontsize',14)
%% Check prs/fr results and plot
all_amps = [];

addpath(fullfile('..','simulation_code','BrewerMap'))
for n_neur = 1:length(neuron_data)
    all_amps = [all_amps [neuron_data(n_neur).amp.amp]]
end
figure(11);histogram(all_amps,20);
group = @(x,xmin,xmax) (x <= xmax) & (x > xmin);
% @g2 = (all_amps >= 55) & (all_amps < 100);
% @g3 = (all_amps >= 100) & (all_amps < 150);
% @g4 = (all_amps >= 150) & (all_amps < 200);
% @g5 = (all_amps > 200);

%divs = [0 55; 55 100; 100 120; 120 150; 150  180; 180 220; 220 500];
%divs = [0 55; 55 100; 100 150; 150  180; 180 220; 220 500];
divs = [0 75; 75 120; 120 175; 175 210; 210 500];%[0 55; 55 100; 100  180; 180 220; 220 500];
col_shift  = 4;
col_dir = brewermap(size(divs,1)+col_shift,'BuPu');
%copper(size(divs,1));
for n_f = 1:size(divs,1)
    figure(1); plot(n_f,n_f,'x','color',col_dir(n_f+col_shift,:)); hold on;
end

%
tmp = get(0, 'Screensize');
  set(gcf, 'Position', tmp )
  title('Real Data')
for n_neur = 1:length(neuron_data)
    fit_order =[];
    for n_amp = 1:length(neuron_data(n_neur).amp)
        cur_amp = neuron_data(n_neur).amp(n_amp).amp;
        amp_grp = [];
        for n_grp = 1:size(divs,1)
            amp_grp = [amp_grp group(cur_amp,divs(n_grp,1), divs(n_grp,2))];
        end
        
        amp_grp = find(amp_grp);       
        fit_order = [fit_order amp_grp];

        figure(12); 
        subplot(2,3,n_neur);
        errorbar([0; neuron_data(n_neur).amp(n_amp).pr(:,1)],...
            [neuron_data(n_neur).amp(n_amp).base_fr(1); neuron_data(n_neur).amp(n_amp).fr(:,1)],...
            [neuron_data(n_neur).amp(n_amp).base_fr(2); neuron_data(n_neur).amp(n_amp).fr(:,2)],...
            '.-','color',col_dir(amp_grp+col_shift,:,:),'markersize',8); hold on;
            figure(13); 
        subplot(2,3,n_neur);
        errorbar([0; neuron_data(n_neur).amp(n_amp).pr(:,1)],...
            neuron_data(n_neur).amp(n_amp).base_fr(1)+[0; neuron_data(n_neur).amp(n_amp).dfr(:,1)],...
            neuron_data(n_neur).amp(n_amp).base_fr(2)+[0; neuron_data(n_neur).amp(n_amp).dfr(:,2)],...
            'o-','color',col_dir(amp_grp+col_shift,:,:),'linewidth',2); hold on;
        xlabel('Pulse Rate (pps)'); ylabel('Firing Rate (sps)')
        ylim([0 200]); title(['Afferent ' num2str(cur_amp_neurs(n_neur))])
        box off;set(gca,'fontsize',14)
        
    end
    
    if (n_neur == length(neuron_data))
        for n_amp = 1:size(divs,1)
            a(n_amp) = plot(0,100,'.','color',col_dir(n_amp+col_shift,:));
        end
    end
    
    neuron_data(n_neur).fit_ord = fit_order;
end
%
  
for n_neur = 1:6
 if (n_neur == length(neuron_data))
        for n_amp = 1:size(divs,1)
            a(n_amp) = plot(0,100,'.','color',col_dir(n_amp+col_shift,:));
        end
    end
end 
    neuron_data(n_neur).fit_ord = fit_order;
legend(a,num2str(divs));


% Predict the data with prediction algorithm
%(1) try fit the best one and the current amp
%(2) Fit in increaseing order from first point only trying higher current
%and minimizing first current
cd(base_dir)

%%
disp('Starting Predictions ...')
%figure(43);
%divs = [0 55; 55 100; 100 150; 150 200; 200 220; 220 1e3];
%col_dir =jet(size(divs,1));

best_fit_Is=  nan(6,5);
best_fit_rms=  nan(6,5);
best_fit_rms_avg_std=  nan(6,5,2);
cur_range = [0 300];
for n_neur = 1:length(neuron_data)
  %figure(100 + n_neur);
    figure(714);
    amp_order = [];
      subplot(2,3,n_neur);
    for n_amp = 1:length(neuron_data(n_neur).amp)
        cur_amp = neuron_data(n_neur).amp(n_amp).amp;
        amp_grp = [];
        for n_grp = 1:size(divs,1)
            amp_grp = [amp_grp group(cur_amp,divs(n_grp,1), divs(n_grp,2))];
        end
        amp_grp = find(amp_grp);
        amp_order = [amp_order amp_grp];
        
        %subplot(5,1,n_amp)
       
        s_tots = vertcat(neuron_data(n_neur).amp.base_fr);
       
        base_fr = neuron_data(n_neur).amp(n_amp).base_fr;
        tmp_data.fr_n = neuron_data(n_neur).amp(n_amp).fr;
        tmp_data.pr_n = neuron_data(n_neur).amp(n_amp).pr;
        n_amps = []; % use for making subplot
            
        errorbar([0; tmp_data.pr_n(:,1)],[base_fr(1); tmp_data.fr_n(:,1)],...
           [base_fr(2); tmp_data.fr_n(:,2)],'.', 'color',col_dir(amp_grp+col_shift,:,:),'linewidth',1,'markersize',8); hold on;

       
        [I_idx,rms_best,fr_pred_best,mean_std_best] = two_d_rms_eval(base_fr(:,1), tmp_data.pr_n(:,1)',tmp_data.fr_n);
        [short_pred] = interp_pred_f_5_5_21(I_idx,base_fr(:,1),tmp_data.pr_n(:,1)');
         [full_pred] = interp_pred_f_5_5_21(I_idx,base_fr(:,1),[0:350]);
        
        best_fit_Is(n_neur,n_amp) = I_idx;
        best_fit_rms(n_neur,n_amp) = rms_best;
        best_fit_rms_avg_std(n_neur,n_amp,:) = mean_std_best;
        plot([0; tmp_data.pr_n(:,1)],base_fr(:,1) + [0 short_pred],'o','color',col_dir(amp_grp+col_shift,:,:));
        plot([0:350],base_fr(:,1) +full_pred,':','color',col_dir(amp_grp+col_shift,:,:),'linewidth',1);  

      %  rms_err(n_neur,n_amp) = rms([base_fr(1); tmp_data.fr_n(:,1)] - (base_fr(:,1) + [0 short_pred])')
     
       % plot([0:300],full_pred+base_fr(:,1),'m--');

%         %ylim([0 200]);
        title(['Afferent ' num2str(cur_amp_neurs(n_neur))]);
        set(gca,'fontsize',14)       
    end
    disp(sprintf('Done with Afferent %d ...',(n_neur)));
   % pause
%     k = [neuron_data(n_neur).amp(:).amp];
%     [amp_order; k ]
 %   suptitle('Predictions')
end

% Current real v. best fit:
figure(406);
col_n = brewermap(6,'Set2');
for n_neur = 1:6
   %subplot(2,3,n_neur)
    tmp =[neuron_data(n_neur).amp.amp];
 [x_ord y_ord]=   sort(tmp);
 plot(x_ord,best_fit_Is(n_neur,y_ord),'o-','color',col_n(n_neur,:),'linewidth',2); hold on;
 %neuron_data(n_neur).best_fit_I(y_ord),'o-','color',col_n(n_neur,:),'linewidth',2); hold on;
xlabel('Real I'); ylabel('Pred I')
a(n_neur) = plot(0,400,'.','color',col_n(n_neur,:),'linewidth',2);


squeeze(best_fit_rms_avg_std(n_neur,y_ord,:))
end
legend(a,{'1','2','3', '4', '5','6'}); box off;
set(gca,'fontsize',15)



% %% 3d plot real data v Convention
% %tmp_Is from fin_plotter
% all_Ps = neuron_data(n_neur).amp(n_amp).pr(:,1);
%     %Compare ideal pulse case to reality: 
%     base_fr=  25;
%     sim_pulse_rule = repmat(base_fr+all_Ps' ,[1 size(tmp_Is,2)]).*linspace(1, 0,1501);
%     sim_pulse_rule2 = base_fr+repmat(all_Ps',[1 size(tmp_Is,2)]).*[ones(1,1001) zeros(1,500)];
%     figure(10);
%     for n_currs = 1:length(collision_sim_results_2)
%         fr_x_currs(n_currs,:) = mean(collision_sim_results_2(n_currs).fr);
%     end
% 
%     
%     hold on;
%     s3 =  surf(tmp_Is*curr_corr*-1000,all_Ps,sim_pulse_rule2);
%     s3.EdgeColor= 'none'; s3.FaceColor = 'r'; s3.FaceAlpha = .3;
%     xlabel('I_{stim} (mA)'); ylabel('Pulse Rate (pps)'); zlabel('fr (sps)');
%     set(gca,'fontsize',20);  
%     
% 
%     for n_neur = 1:6
%         for n_amp  = 1:length(neuron_data(n_neur).amp)
%             plot3(repmat(neuron_data(n_neur).amp(n_amp).amp,...
%                 [1 length(neuron_data(n_neur).amp(n_amp).pr(:,1))+1])',...
%                 [0; neuron_data(n_neur).amp(n_amp).pr(:,1)], [neuron_data(n_neur).amp(n_amp).base_fr(1); neuron_data(n_neur).amp(n_amp).fr(:,1)],'o-')
%         end
%     end
%     
%     %%
%     figure(11);
%     for n_neur = 1:6
%         subplot(2,3,n_neur)
%         prs = []; frs = []; Is = [];
%         for n_amp  = 1:length(neuron_data(n_neur).amp)
%             prs = [prs; neuron_data(n_neur).amp(n_amp).pr(:,1)];
%             frs = [frs; neuron_data(n_neur).amp(n_amp).fr(:,1)];
%             Is = [Is; repmat(neuron_data(n_neur).amp(n_amp).amp,[length(neuron_data(n_neur).amp(n_amp).pr(:,1)) 1])];
%         end
%         
%           I_range =min(Is):5:max(Is);
%         pr_range=  min(prs):5:max(prs);
%           pred_z = griddata(Is,prs,frs,I_range,pr_range');
%       
%      
%             s4= surf(I_range,pr_range,pred_z); hold on;%,'b'); hold on;
%               s3 =  surf(tmp_Is*curr_corr*-1000,all_Ps,sim_pulse_rule2);
%     s3.EdgeColor= 'none'; s3.FaceColor = 'r'; s3.FaceAlpha = .3;
% 
%       %  xlabel('I (uA)'); ylabel('Pulse Rate (pps)'); zlabel('Firing Rate (sps)')
%         set(gca,'fontsize',14)
%     end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS FOR ABOVE
function [fin_nums base_fr] = data_split(data_per_aff,plot_it,cur_neur,aff_num,n_neurons,afferent_info,n_plot)
pulse_timing = find(data_per_aff.stim);
pulse_block_div = find(diff(pulse_timing) > 250);
pulse_block_start = [pulse_timing(1); pulse_timing(pulse_block_div+1)];
pulse_block_end = [pulse_timing(pulse_block_div); pulse_timing(end)];

single_spk = (pulse_block_start == pulse_block_end);
pulse_block_start(single_spk) = [];
pulse_block_end(single_spk) = [];
if plot_it
    %For each amplitude per neuron
    %subplot(3,1,1)

    plot(data_per_aff.stim); hold on;
    plot(pulse_block_start,data_per_aff.stim(pulse_block_start),'r.'); hold on;
    plot(pulse_block_end,data_per_aff.stim(pulse_block_end),'b.');
    xlabel('Time (ms)')
    %pause; %cla %%%%Note afferent 3 had weird block spacing
end

%Get data per block
clear base_firing_btw_block
min_inter_wind_length = 300;%before 1000 ms
for n_p = 1:length(pulse_block_start)
    p_rate_per_block(n_p) = sum((pulse_timing >= pulse_block_start(n_p)) & ...
        (pulse_timing <= pulse_block_end(n_p)))/(1e-3*(pulse_block_end(n_p) - pulse_block_start(n_p)));
    firing_rate_per_block(n_p) = sum((find(data_per_aff.ua) >= pulse_block_start(n_p)) & ...
        (find(data_per_aff.ua) <= pulse_block_end(n_p)))/(1e-3*(pulse_block_end(n_p) - pulse_block_start(n_p)));
   
    %Extra information for window testing:
    n_pulse(n_p) = sum((pulse_timing >= pulse_block_start(n_p)) & ...
        (pulse_timing <= pulse_block_end(n_p)));
    n_aps(n_p) =  sum((find(data_per_aff.ua) >= pulse_block_start(n_p)) & ...
        (find(data_per_aff.ua) <= pulse_block_end(n_p)));
    

    %Estimate base firing rate during the trial:
    %Only use up to a 1000 ms before data to avoid too close to last trial:
    %Don't use data within 100 ms of the last trial end
    %max_wind = 5000; %ms
    dist_last = 50; %ms
    if n_p ==1
        if pulse_block_start(n_p) < min_inter_wind_length %ms
            % Have no data like this
            warning(sprintf('Aff %s block %s issue',num2str(aff_num),num2str(n_p)));
            base_firing_btw_block(n_p) = nan;
        else
            %1000 ms max
%             base_firing_btw_block(n_p) = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
%               (find(data_per_aff.ua) >= (pulse_block_start(n_p) - max_wind)))/(1e-3*(pulse_block_start(n_p)));
            base_firing_btw_block(n_p) = sum(find(data_per_aff.ua) < pulse_block_start(n_p))/(1e-3*(pulse_block_start(n_p)));
        end
    else
        if (pulse_block_start(n_p) - pulse_block_end(n_p-1)) < min_inter_wind_length
             warning(sprintf('Aff %s block %s issue',num2str(aff_num),num2str(n_p)));
            last_idx = max(find(~isnan(base_firing_btw_block(1:(n_p-1)))));
            cur_idx  = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
                (find(data_per_aff.ua) > ...
               (pulse_block_end(n_p-1)+ dist_last))) ...
                /(1e-3*(pulse_block_start(n_p) - pulse_block_end(n_p-1)));
            base_firing_btw_block(n_p) = mean([base_firing_btw_block(last_idx),cur_idx]);%nan;
            % Consider averaging in with past data instead of this (try
            % this first)
        else
%             base_firing_btw_block(n_p) = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
%                 (find(data_per_aff.ua) > ...
%                 max((pulse_block_end(n_p-1)+ dist_last),pulse_block_start(n_p) - max_wind))) ...
%                 /(1e-3*(pulse_block_start(n_p) - pulse_block_end(n_p-1)));
 base_firing_btw_block(n_p) = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
                (find(data_per_aff.ua) > ...
               (pulse_block_end(n_p-1)+ dist_last))) ...
                /(1e-3*(pulse_block_start(n_p) - pulse_block_end(n_p-1)));
        end
    end
end
base_fr = [mean(base_firing_btw_block) std(base_firing_btw_block)];
dfr = firing_rate_per_block - mean(base_firing_btw_block);

 [coeff] =best_fit_line(p_rate_per_block,dfr);
 
 fin_nums.slope = coeff.Coefficients.Estimate;
% Print out info:
verbose = 0;
if verbose
   disp(sprintf('Aff %s ',num2str(aff_num)));
    clear len_pre_wind
    len_pr_wind = pulse_block_end -  pulse_block_start;
    len_pre_wind(1) = pulse_block_start(1);
    len_pre_wind(2:(length(pulse_block_start))) = pulse_block_start(2:end) - pulse_block_end(1:(end-1));
    
    disp('Wind lengths PR/btw (s); N P AP in PR wind; PR FR in wind')
    disp([round(len_pr_wind/1e3,3), round(len_pre_wind/1e3,3)',n_pulse',n_aps',round(p_rate_per_block,3)',round(firing_rate_per_block,3)'])
end
 
%Combine across same pulse rate: - make sure only getting one pr per pr
%group
unique_ps = unique(round(p_rate_per_block));
un_ps = [];
for n_u_ps = 1:length(unique_ps)
    if ~(sum(abs(unique_ps(n_u_ps) - round(unique_ps)) < 3) > 1)
       
        un_ps = [un_ps unique_ps(n_u_ps)];  
    end
end
 
%un_ps;
unique_ps = un_ps; un_ps = [];
for n_unique_ps = 1:length(unique_ps)
    pr_n(n_unique_ps).idxs = find(abs(unique_ps(n_unique_ps) - round(p_rate_per_block)) < 3);
    fin_nums.pr(n_unique_ps,:) = [mean(round(p_rate_per_block(pr_n(n_unique_ps).idxs))) ...
        std(round(p_rate_per_block(pr_n(n_unique_ps).idxs)))];
    fin_nums.fr(n_unique_ps,:) = [mean(round(firing_rate_per_block(pr_n(n_unique_ps).idxs))) ...
        std(round(firing_rate_per_block(pr_n(n_unique_ps).idxs)))];
    fin_nums.dfr(n_unique_ps,:) = [mean(round(dfr(pr_n(n_unique_ps).idxs))) ...
        std(round(dfr(pr_n(n_unique_ps).idxs)))];
end

   
    figure(n_plot); subplot(2,3,cur_neur);%subplot(n_neurons,1,cur_neur);
    errorbar([0; fin_nums.pr(:,1)],[base_fr(:,1); fin_nums.fr(:,1)],...
        [base_fr(:,2); fin_nums.fr(:,2)],'.-');hold on; 
    
    xlabel('Pulse Rate'); ylabel('Firing Rate'); box off;
    title(sprintf('Afferent %s: CV*: %s',num2str(aff_num), num2str(round(afferent_info(aff_num,5),2))));

end


%%% ANALYSES MITCHELL ET AL. DATA!
function [fin_nums base_fr block_dat] = data_t_fr(data_per_aff,plot_it,cur_neur,aff_num,n_neurons,afferent_info,n_plot)
pulse_timing = find(data_per_aff.stim);
pulse_block_div = find(diff(pulse_timing) > 250);
pulse_block_start = [pulse_timing(1); pulse_timing(pulse_block_div+1)];
pulse_block_end = [pulse_timing(pulse_block_div); pulse_timing(end)];

single_spk = (pulse_block_start == pulse_block_end);
pulse_block_start(single_spk) = [];
pulse_block_end(single_spk) = [];
if plot_it
    figure(1); %For each amplitude per neuron
    subplot(3,1,1)
    plot(data_per_aff.stim); hold on;
    plot(pulse_block_start,data_per_aff.stim(pulse_block_start),'r.'); hold on;
    plot(pulse_block_end,data_per_aff.stim(pulse_block_end),'b.');
    xlabel('Time (ms)')
    %pause; %cla %%%%Note afferent 3 had weird block spacing
end

%Get data per block
min_inter_wind_length = 300;%500;%before 1000 ms
for n_p = 1:length(pulse_block_start)
    
    p_rate_per_block(n_p) = sum((pulse_timing >= pulse_block_start(n_p)) & ...
        (pulse_timing <= pulse_block_end(n_p)))/(1e-3*(pulse_block_end(n_p) - pulse_block_start(n_p)));
    firing_rate_per_block(n_p) = sum((find(data_per_aff.ua) >= pulse_block_start(n_p)) & ...
        (find(data_per_aff.ua) <= pulse_block_end(n_p)))/(1e-3*(pulse_block_end(n_p) - pulse_block_start(n_p)));
    %Estimate base firing rate during the trial:
    if n_p ==1
        if pulse_block_start(n_p) < min_inter_wind_length %ms
            base_firing_btw_block(n_p) = nan;
        else
            base_firing_btw_block(n_p) = sum(find(data_per_aff.ua) < pulse_block_start(n_p))/(1e-3*(pulse_block_start(n_p)));
        end
    else
        if (pulse_block_start(n_p) - pulse_block_end(n_p-1)) < min_inter_wind_length
            last_idx = max(find(~isnan(base_firing_btw_block(1:(n_p-1)))));
            base_firing_btw_block(n_p) = base_firing_btw_block(last_idx);%nan;
        else
            base_firing_btw_block(n_p) = sum((find(data_per_aff.ua) < pulse_block_start(n_p)) & ...
                (find(data_per_aff.ua) > pulse_block_end(n_p-1)))/(1e-3*(pulse_block_start(n_p) - pulse_block_end(n_p-1)));
        end
    end
end
%Estimate mean and std across trial
base_fr = [nanmean(base_firing_btw_block) nanstd(base_firing_btw_block)];

block_dat.pr = p_rate_per_block;
block_dat.fr = firing_rate_per_block;
block_dat.base_fr = base_firing_btw_block;
sub_base = block_dat.base_fr;
%sub_base(isnan(block_dat.base_fr)) = mean(sub_base(~isnan(block_dat.base_fr)));
dfr = block_dat.fr - sub_base;


%% Fit a slope to increase with each current:
 [coeff] =best_fit_line(block_dat.pr,dfr);
 
% %  figure(1); plot(block_dat.pr,dfr,'k.'); hold on;
% %  plot([0:300],coeff.Coefficients.Estimate*[0:300],'r')
 
 fin_nums.m_per_amp = coeff.Coefficients.Estimate;
%%
if plot_it
%figure(2);
subplot(3,1,2);%subplot(n_neurons,1,cur_neur);
plot(1:length(pulse_block_start),p_rate_per_block,'.'); hold on;
xlabel('Block Num'); ylabel('Pulse Rate');
end

%Combine across same pulse rate:
unique_ps = unique(round(p_rate_per_block));


for n_unique_ps = 1:length(unique_ps)
    pr_n(n_unique_ps).idxs = find(abs(unique_ps(n_unique_ps) - round(p_rate_per_block)) < 2);
    fin_nums.pr_n(n_unique_ps,:) = [mean(round(p_rate_per_block(pr_n(n_unique_ps).idxs))) ...
        std(round(p_rate_per_block(pr_n(n_unique_ps).idxs)))];
    fin_nums.fr_n(n_unique_ps,:) = [mean(round(firing_rate_per_block(pr_n(n_unique_ps).idxs))) ...
        std(round(firing_rate_per_block(pr_n(n_unique_ps).idxs)))];
    fin_nums.dfr_n(n_unique_ps,:) = [mean(round(dfr(pr_n(n_unique_ps).idxs))) ...
        std(round(dfr(pr_n(n_unique_ps).idxs)))];
end

%if plot_it
    figure(n_plot); subplot(2,3,cur_neur)%subplot(n_neurons,1,cur_neur);
    errorbar([0; fin_nums.pr_n(:,1)],[base_fr(:,1); fin_nums.fr_n(:,1)],...
        [base_fr(:,2); fin_nums.fr_n(:,2)],'k.');
    hold on; plot([0; fin_nums.pr_n(:,1)],[base_fr(:,1); fin_nums.fr_n(:,1)],'k');
    xlabel('Pulse Rate'); ylabel('Firing Rate'); box off;
    title(sprintf('Afferent %s: CV*: %s',num2str(aff_num), num2str(round(afferent_info(aff_num,5),2))));
%end
end



function [coeff] =best_fit_line(prs,dfrs)
coeff = fitlm(prs,dfrs,'Intercept',false);
end