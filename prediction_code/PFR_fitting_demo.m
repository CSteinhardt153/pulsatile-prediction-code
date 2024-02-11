%PFR_fitting_demo.m
%Fit data from simulations at each Spontaneous rate of afferent (S) and for
%each pulse amplitude(I) and pulse rate(pr) combination
%>Loads in simulation data and pre-chosen hyperparameter starting points
%for doing optimization of predictive equations
%>Applicable for similar situation with other neuron types when data on
%firing rate for pulse rate, pulse amplitude, and spontaneous rate is known
%and when goal is to characterize pulse parameter firing rate relationship.
%>>Uses include finding equivalent rules in other neural system or
%adding this code to neural implants using pulses as a correction.

% Copyright (c) 2022  The Johns Hopkins University
%All rights reserved. Redistribution and use in source and binary forms,
% with or without modification, are permitted provided that the following 
% conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright 
% notice, this list of conditions and the following disclaimer in the 
% documentation and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its 
% contributors may be used to endorse or promote products derived from this
% software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
% “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
% TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%==========================================================
cd('..')
base_dir = pwd;
addpath(genpath(fullfile(base_dir,'helper_functions')));
data_dir = fullfile(base_dir,'relevant_data');
%Seeded simulation data with 10 repetitions
%First file point to origin of the simulations that are compressed into a nicer format in
%pr_fr_S* files
load(fullfile(data_dir,'seed_sims_R10_01_29_22.mat'),'Is','prs','file_names');
perS_dat = dir(fullfile(data_dir,'pr_fr_S*_01-15-2023.mat'))
 
 for n_S = 1:length(perS_dat)
   tmp = strsplit(perS_dat(n_S).name,'_');
   dat_S(n_S) = str2num(tmp{3}(2:end));
 end
[ord ord_S_idx] = sort(dat_S);

perS_dat = perS_dat(ord_S_idx);

%Optimization parameter starting points:
 load(fullfile(data_dir,'optimization_seed_params.mat'))
 s_fits = optstart_hyper_params.s_fits; %More information about hyperparameters in this file
%%
 col_s0 = winter(7);
 
 %I: Fitting subset of simulated data per case where hyperparameters
 %initialized with handfit estimates of best guesses.

for n_S = 1:size(s_fits,1) %f = 1:length(file_names)

    load(fullfile(data_dir,perS_dat(n_S).name),'per_S','rel_files');
    S = mean(per_S.pr_fr_dat(:,1,1))
    
    %Find relevant fit
    x_fits = s_fits{n_S}(:,3:end); % first rows are S and I
    I_fits = s_fits{n_S}(:,2);
    n_I_fits = size(x_fits,1);
    S = mean(s_fits{n_S}(:,1));%?mean(rel_s(:,1));
    
    for n_Is = 1:n_I_fits
        x_fit = x_fits(n_Is,:);
        n = find(ismember(Is,I_fits(n_Is)));%index in data
        if isempty(n)
            break
        end
        diff_fr =  squeeze(mean(per_S.pr_fr_dat(:,:,n),1) - mean(per_S.pr_fr_dat(:,:,1)));%squeeze(mean(pr_fr_per_S(n_S,:,n,:),4)) - squeeze(mean(pr_fr_per_S(n_S,:,1,:),4));
        diff_fr_reps = squeeze(per_S.pr_fr_dat(:,1:175,n) - per_S.pr_fr_dat(:,1:175,1));
        
        S = mean(mean(per_S.pr_fr_dat(:,:,1),2));
        %Outlier removal section:
        the_outliers = (isoutlier(diff_fr,"movmedian",55));
        idx_outlier = find(the_outliers);
        prs_fin = 0:410;%prs;%(~ismember(prs,ignore_prs));
        prs_fin(prs_fin==0)= 1;
        
        if sum(the_outliers) == 0
            frs_real = diff_fr;%end-10);%~the_outliers);
            use_idx = 1:length(diff_fr);
        else
            
            frs_real = diff_fr;
            for n_out = 1:length(idx_outlier)
                av_idx = min(max(1,(idx_outlier(n_out)-8):(idx_outlier(n_out)+8)),length(frs_real));
                av_idx = av_idx(~ismember(av_idx,idx_outlier));
                frs_real(idx_outlier(n_out)) = mean(frs_real(av_idx));
                
            end
        end
        
         prs= per_S.pulse_rate(1:175);
         frs_real = frs_real(1:175);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Fit Equations:
        prt_1 = @(a,p) min(a.*p,p);
        prt_2 = @(d,b,p,S) max(-S,d*b.*p);%max(-d*S,d*b.*p); %took out the d 4/14 for checking
        prt_3 = @(c,p,S,e) max(-S,min(0,c.*(p-e)));
        
        %New rule with scale up as a parameters
        pp_ps_pred_new = @(a,b,c,d,tpp,prt_elim1,prt_elim2,p,e,scale_ups) ...
            max(-S,d*(part_elim_calc(tpp,Is(n),p,S,[prt_elim1 prt_elim2],scale_ups)) ... %)*(a<0.02 | a > 0.8)...
            + prt_2(d,b,p,S) + prt_3(c,p,S,e)+ min(a.*p,p)) ;%*(                 
            
        if (S==0)
            
            fac_fun = @(p,p_cut,sig_scale) 1./(1+exp(sig_scale.*(-p + p_cut))); %.2 was working
            
            %Standard equatiosn including the facil part
            error_fun_handfit = @(a,b,c,d,t_pp,prt_elim1,prt_elim2,frs_pred,p,e,scale_up1,scale_up2,p_facil,sig_scale) ...
                mse(frs_pred -  fac_fun(prs,p_facil,sig_scale).*pp_ps_pred_new(a,b,c,d,t_pp,prt_elim1,...
                prt_elim2,p,e,[scale_up1 scale_up2])); %error function
            
            fun = @(x) error_fun_handfit(x(1),x(2),x(3),x(4),x(5),x(6),x(7),...
                frs_real,prs,x(8),x(9),x(10),x_fit(11),x_fit(12));
            
            auto_fit =  fac_fun(prs,x_fit(11),x_fit(12)).*pp_ps_pred_new(x_fit(1),x_fit(2),x_fit(3),x_fit(4),...
                x_fit(5),x_fit(6),x_fit(7),prs,x_fit(8),[x_fit(9) x_fit(10)]);
            
            %Try optimizer:
            lb = [0    -1  -1   0 1     0    0 0   0 0 -1e4 0];%              1                0.01   0.01];
            ub = [1     0    0  1 600  .99 .99 500 500 500 600 1];
            x0 = x_fit;
            
            A = []; b= []; Aeq= [ ]; beq = [];
            [x_fit_opt err_opt]= patternsearch(fun, x0, A,b, Aeq,beq,lb,ub,[]);
            opt_fit = fac_fun(prs,x_fit_opt(11),x_fit_opt(12)).*...
                pp_ps_pred_new(x_fit_opt(1),x_fit_opt(2),x_fit_opt(3),x_fit_opt(4),...
                x_fit_opt(5),x_fit_opt(6),x_fit_opt(7),prs,x_fit_opt(8),[x_fit_opt(9) x_fit_opt(10)]);
            
        else
            
            error_fun_handfit = @(a,b,c,d,t_pp,prt_elim1,prt_elim2,frs_pred,p,e,scale_up1,scale_up2) ...
                mse(frs_pred - pp_ps_pred_new(a,b,c,d,t_pp,prt_elim1,prt_elim2,p,e,[scale_up1 scale_up2])); %error function
            
            fun = @(x) error_fun_handfit(x(1),x(2),x(3),x(4),x(5),x(6),x(7),...
                frs_real,prs,x(8),x(9),x(10));%perS.t(1).pr_fr(n,:)-perS.t(1).pr_fr(1,:))
            
            
            auto_fit =  pp_ps_pred_new(x_fit(1),x_fit(2),x_fit(3),x_fit(4),...
                x_fit(5),x_fit(6),x_fit(7),prs,x_fit(8),[x_fit(9) x_fit(10)]);
            
            %Try optimizer:
            lb = [0    -1  -1   0 1     0    0 0   0 0];%              1                0.01   0.01];
            ub = [1     0    0  1 600  .99 .99 500 500 500];
            x0 = x_fit;
            
            A = []; b= []; Aeq= [ ]; beq = [];
            [x_fit_opt err_opt]= patternsearch(fun, x0, A,b, Aeq,beq,lb,ub,[]);
            opt_fit = pp_ps_pred_new(x_fit_opt(1),x_fit_opt(2),x_fit_opt(3),x_fit_opt(4),...
                x_fit_opt(5),x_fit_opt(6),x_fit_opt(7),prs,x_fit_opt(8),[x_fit_opt(9) x_fit_opt(10)]);
            
        end
        
        fits(n_S).hand_err(n_Is) = fun(x_fit);
        fits(n_S).opt_err(n_Is) = fun(x_fit_opt);
        fits(n_S).x_fit_opts(n_Is,:) = x_fit_opt(1:10);
        fits(n_S).x_fit_h(n_Is,:) = x_fit(1:10);
        if S==0
           fits(n_S).x_fit_fac(n_Is,:) = x_fit(11:12);
           fits(n_S).x_fit_fac_o(n_Is,:) = x_fit_opt(11:12);
          
        end
        
        err = fits(n_S).opt_err;
        %PLOTTING RESULTS:
        figure(1);
        subplot(5,4,n_Is)
        %plot(prs, diff_fr,'k'); hold on;
        shadedErrorBar(prs,mean(diff_fr_reps,1),std(diff_fr_reps,[],1)/sqrt(10),...
            'lineProps',{'-','color',  col_s0(n_S,:)}); hold on;
 
       % plot(prs,frs_real,'k.-','color',[0 0 0 0.9],'linewidth',2); hold on;
        plot(prs,auto_fit,'r','linewidth',1); 
        plot(prs,opt_fit,'g','linewidth',1); ylim([min(frs_real)-10 20+max(frs_real)])
        title(sprintf('I: %0.1f S: %0.1f rms= %0.3f',Is(n), S,err(n_Is)));

    end
    tot_err= fits(n_S).opt_err; 
    disp(sprintf('For S=%0.2f sps, rms = %0.2f +- %0.2f sps',S,mean(tot_err),std(tot_err)));
end
 

%% II: Fitting/ Prediction on all the different amplitudes left out and not:
tic
for n_S = 1:7

    load(fullfile(data_dir,perS_dat(n_S).name),'per_S','rel_files');
    S = mean(per_S.pr_fr_dat(:,1,1))
    
    %Find relevant fit
    x_fits = s_fits{n_S}(:,3:end);
    I_fits = s_fits{n_S}(:,2);
    n_I_fits = size(x_fits,1);
    
    n_p = 1;
    parfor n_Is = 1:44

        I= Is(n_Is);
     
        diff_fr =  squeeze(mean(per_S.pr_fr_dat(:,:,n_Is),1) - mean(per_S.pr_fr_dat(:,:,1)));%squeeze(mean(pr_fr_per_S(n_S,:,n,:),4)) - squeeze(mean(pr_fr_per_S(n_S,:,1,:),4));
        diff_fr_reps = squeeze(per_S.pr_fr_dat(:,1:175,n_Is) - per_S.pr_fr_dat(:,1:175,1));
        
        S = mean(mean(per_S.pr_fr_dat(:,:,1),2));
        %Outlier removal section:
        the_outliers = (isoutlier(diff_fr,"movmedian",55));
        idx_outlier = find(the_outliers);
      
        if sum(the_outliers) == 0
            frs_real = diff_fr;
            use_idx = 1:length(diff_fr);
        else
            
            frs_real = diff_fr;
            for n_out = 1:length(idx_outlier)
                av_idx = min(max(1,(idx_outlier(n_out)-8):(idx_outlier(n_out)+8)),length(frs_real));
                av_idx = av_idx(~ismember(av_idx,idx_outlier));
                frs_real(idx_outlier(n_out)) = mean(frs_real(av_idx));
                
            end
        end
        prs= per_S.pulse_rate(1:175);
        frs_real = frs_real(1:175);
               
        [err_o,err_int,opt_fit,interp_fit,x_fit_opt,x_fit] = fit_pr_fr_data(S,I,prs,frs_real);
        
        %cla;
%         figure(1); 
%         subplot(2,5,n_p)
%         shadedErrorBar(prs,mean(diff_fr_reps,1),std(diff_fr_reps,[],1)/sqrt(10),...
%             'lineProps',{'-','color',  col_s0(n_S,:)}); hold on;
%         plot(prs,interp_fit,'r');
%         plot(prs,opt_fit,'g');
%         title(num2str(I));
%         pause(0.1)
        
        n_p = n_p+1;
        disp(sprintf('S: %0.1f I:%0.1f Opt: %0.2f Int:%0.2f',S,I,sqrt(err_o),sqrt(err_int)));
        
        rms_err_o(n_Is) = sqrt(err_o);
        rms_err_int(n_Is) = sqrt(err_int);
        opt_fit(n_Is,:) = (opt_fit);
        interp_fit(n_Is,:) = (interp_fit);
        opt_params(n_Is,:) = x_fit_opt(1:10);
        interp_params(n_Is,:) = x_fit(1:10);
        if S==0
        opt_fac_params(n_Is,:) = x_fit_opt(11:12);
        interp_fac_params(n_Is,:) = x_fit(11:12);
        end
        %Originally save and and not tested in parallel mode:
        % fits.rms_err_o(n_S,n_Is) = sqrt(err_o);
        % fits.rms_err_int(n_S,n_Is) = sqrt(err_int);
        % fits.opt_fit(n_S,n_Is,:) = (opt_fit);
        % fits.interp_fit(n_S,n_Is,:) = (interp_fit);
        % fits.opt_params(n_S,n_Is,:) = x_fit_opt(1:10);
        % fits.interp_params(n_S,n_Is,:) = x_fit(1:10);
        % if S==0
        % fits.opt_fac_params(n_S,n_Is,:) = x_fit_opt(11:12);
        % fits.interp_fac_params(n_S,n_Is,:) = x_fit(11:12);
        % end
        
    end
    fit_all(n_S).rms_err_o= rms_err_o;
    fit_all(n_S).rms_err_int = rms_err_int;
    fit_all(n_S).opt_fit = opt_fit;
    fit_all(n_S).interp_fit = interp_fit;
    fit_all(n_S).opt_params = opt_params;
    fit_all(n_S).interp_params = interp_params;
    if S==0
        fit_all(n_S).opt_fac_params
        fit_all(n_S).interp_fac_params 
    end
end
toc
%code_file = 'I_S_fit_2_21_23.m' - oringal name of this file and its output;
%save('S_I_F_fits_2_24_23.mat','fits','perS_dat','code_file')%%%%
%save('test_nodS_S_I_F_fits_4_14_23.mat','fits','perS_dat','code_file')

%% Plot fits after optimizing:
%Figure 4 A
load(fullfile(data_dir,'S_I_F_fits_2_24_23.mat'))
view_Is =[37 38 41 44] %round(linspace(1,44,12));%[10 12 15 20]%round(linspace(1,44,16) %For visualizing examples
col_s0 = winter(7);
prs= per_S.pulse_rate(1:175);

for n_S = 1:7%
    
    load(fullfile(data_dir,perS_dat(n_S).name),'per_S','rel_files');
    S = mean(per_S.pr_fr_dat(:,1,1));
    S_ord(n_S) = S;
        
    n_p = 1; % number of plots
    
    for n_Is = view_Is
         diff_fr_reps = squeeze(per_S.pr_fr_dat(:,1:175,n_Is) - per_S.pr_fr_dat(:,1:175,1));
        
        figure(2);
        subplot(1,4,n_p)
        if (S - 0) < 1
            shadedErrorBar(prs,mean(diff_fr_reps,1),std(diff_fr_reps,[],1)/sqrt(10),...
                'lineProps',{'-','color', 'k','linewidth',2}); hold on;
        else
            shadedErrorBar(prs,mean(diff_fr_reps,1),std(diff_fr_reps,[],1)/sqrt(10),...
                'lineProps',{'-','color',  col_s0(8-n_S,:)}); hold on;
        end

        plot(prs,squeeze(fits.opt_fit(n_S,n_Is,:)),'color',[1 0 0 0.7],'linewidth',1.5);
        title(['I = ' num2str(per_S.I(n_Is)*-20)]); box off;
       
        if n_p >= (length(view_Is)-2)
           xlabel('Pulse Rate (pps)') ;
        end
        if mod(n_p,4) == 1
            ylabel('Firing Rate (sps)') ;
        end
         n_p = n_p +1;
         set(gca,'fontsize',14)
    end
end
%% Find rms where predicted and didn't predict:
%find rms with each repetition of mean fit:

Is = per_S.I(n_Is)*-20;
rms_f = @(a,b,dim) sqrt(sum((a - b).^2,dim)/size(b,dim));
for n_S = 1:7
    load(fullfile(data_dir,perS_dat(n_S).name),'per_S','rel_files');
    S_dat(n_S) = mean(per_S.pr_fr_dat(:,1,1));

    diff_per_rep = squeeze(per_S.pr_fr_dat(:,1:175,:) - per_S.pr_fr_dat(:,1:175,1));
    rms_per_I(n_S,:,:) =rms_f(diff_per_rep,permute(repmat(squeeze(fits.opt_fit(n_S,:,:))',[1,1,10]),[3 1 2]),2);

end


S_dat_ord = [0 5 13 28 51 80 131];
%rms_per_S =[S_dat_ord' mean(mean(rms_per_I,2),3) std(mean(rms_per_I,2),[],3)/sqrt(10)]
rms_per_S =[S_dat_ord' mean(mean(rms_per_I,2),3) std(mean(rms_per_I,2),[],3)/sqrt(10)]
rms_cross_S =[mean(rms_per_S(:,2)) std(rms_per_S(:,2))/sqrt(7)]
%% Figure 4 B-C Plot RMS results
load(fullfile(data_dir,perS_dat(n_S).name),'per_S','rel_files');
figure(20);histogram(squeeze(std(rms_per_I,[],2)),25,'EdgeColor','k','FaceColor','k'); xlabel('Standard Deviation (sps) ')
ylabel('Pulse rate-Firing rate Relationships')
set(gca,'fontsize',14); box off;

rms_per_I_S = squeeze(mean(rms_per_I,2));
figure(15);

%ID and compare performance

perS_dat = dir(fullfile(data_dir ,'pr_fr_S*_01-15-2023.mat'))
load(fullfile(data_dir,perS_dat(1).name),'per_S')
load(fullfile(data_dir,'optimization_seed_params.mat')) %gets s_fits


for n_S = 1:7
    fitted_Is(n_S,:) = (ismember(per_S.I*-20,s_fits{n_S}(:,2)));
end

boxchart(rms_per_I_S', 'MarkerStyle', 'none')
hold on;
S_dat_ord_str = ['0' '5' '13' '28' '51' '80' '131'];

for n = 1:7

    scatter(.45*(rand(1,sum(fitted_Is(n,:)))-.5)+repmat(n,[1 sum(fitted_Is(n,:))]),rms_per_I_S(n,find(fitted_Is(n,:))),'MarkerEdgeColor',[0 0 1]); hold on;
    scatter(.45*(rand(1,sum(~fitted_Is(n,:)))-.5)+repmat(n,[1 sum(~fitted_Is(n,:))]),rms_per_I_S(n,find(~fitted_Is(n,:))),'MarkerFaceAlpha',0.5,'MarkerEdgeColor',[1 0 0]); hold on;

end

set(gca,'xticklabel', {'0', '5', '13' ,'28' ,'51' ,'80' ,'131'})
ylabel('RMS (sps)')
xlabel('Spontaneous Rate (sps)')
set(gca,'fontsize',15)
%%
%T-test for all of this:
for n_S = 1:7
        [h,p,ci,stats] = ttest2(rms_per_I_S(n_S,find(fitted_Is(n,:))),...
            rms_per_I_S(n_S,find(~fitted_Is(n,:))));
        p_ttest(n_S) = p;
        h_ttest(n_S) = h;
end


%% HAND FITS:
%s_0,s_5,s_13,s_30,s_55,s_80,s_125
s_0 = s_fits{1};s_5 = s_fits{2};
s_13 = s_fits{3};s_30 = s_fits{4}; 
s_55 = s_fits{5};s_80 = s_fits{6};
s_125 = s_fits{7};
sim_Is = 0:10:336;
s_0_vars = interp1(s_0(:,2),s_0(:,3:end),sim_Is);
s_5_vars = interp1(s_5(:,2),s_5(:,3:end),sim_Is);
s_13_vars = interp1(s_13(:,2),s_13(:,3:end),sim_Is);
s_30_vars = interp1(s_30(:,2),s_30(:,3:end),sim_Is);
s_55_vars = interp1(s_55(:,2),s_55(:,3:end),sim_Is);
s_80_vars = interp1(s_80(:,2),s_80(:,3:end),sim_Is);
s_125_vars = interp1(s_125(:,2),s_125(:,3:end),sim_Is);

S_I_vars(1,:,:)=s_0_vars(:,1:10);
S_I_vars(2,:,:)=s_5_vars;
S_I_vars(3,:,:)=s_13_vars;
S_I_vars(4,:,:)=s_30_vars;
S_I_vars(5,:,:)=s_55_vars;
S_I_vars(6,:,:)=s_80_vars;
S_I_vars(7,:,:)=s_125_vars;

Ss = [s_0(1,1) s_5(1,1) s_13(1,1) s_30(1,1)...
    s_55(1,1) s_80(1,1) s_125(1,1)];

var_names = {'m_{facil}','SxP','PxS','p_{AP}','t_b','p_{elim1}','p_{elim2}',...
    'b_{PxS}','k_{elim1}','k_{elim2}'};
for i_var = 1:10
figure(10); subplot(2,5,i_var)
[Xq,Yq] = meshgrid(sim_Is,Ss);
surf(Xq,Yq,abs(squeeze(S_I_vars(:,:,i_var))));
ylabel('S (sps)');xlabel('I (\muA)'); zlabel(var_names{i_var})
if i_var == 5
    zlim([0 40])
end
title(var_names{i_var})
end
% %%
% % OPTIMAL FITS:
% for i_var = 1:10
% figure(20); subplot(2,5,i_var)
% [Xq,Yq] = meshgrid(Is,S_ord);
% surf(Xq,Yq,abs(squeeze(fits.opt_params(:,:,i_var))));
% ylabel('S (sps)');xlabel('I (\muA)'); zlabel(var_names{i_var})
% if i_var == 5
%     zlim([0 40])
% end
% title(var_names{i_var})
% end
