function [err_o,err_int,opt_fit,interp_fit,x_fit_opt,x_fit] = fit_pr_fr_data(S,I,prs,frs_real)
%Outputs:
%error_optimal, error interpolated, opt_fit, interp_fit, optimal params,
%interp_params
%Inputs: S(spont rate), I(pulse amplitude), prs (pulse rate), frs_real
%(real responses for given parameters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_dir = fullfile(pwd,'relevant_data');
%Interpolate:
load(fullfile(data_dir,'optimization_seed_params.mat'))
S_fits = optstart_hyper_params.Ss;
all_fits =  optstart_hyper_params.s_fits; 
[a idxs]=sort(abs(optstart_hyper_params.Ss - S)); %simple interpolation here for parameter fit start

%%%closeS = all_fits{idxs(1)};
%%%sort(abs(closeS))
close1 = all_fits{idxs(1)};
close2 = all_fits{idxs(2)};
%Interp I:
x_fit1= interp1(close1(:,2),close1(:,3:end),I);
x_fit2= interp1(close2(:,2),close2(:,3:end),I);

%Interp S:
x_fit_pred =interp1(S_fits(idxs(1:2)),[x_fit1(1:10);x_fit2(1:10)],S);
x_fit = x_fit_pred;

if S == 0  
   x_p(1) = interp1(all_fits{1}(:,2),all_fits{1}(:,13),I); %s=0 case
   x_p(2) = interp1(all_fits{1}(:,2),all_fits{1}(:,14),I);
   x_fit(11:12) = [x_p];
end

 %Fit Equations:
        prt_1 = @(a,p) min(a.*p,p);
        prt_2 = @(d,b,p,S) max(-d*S,d*b.*p);
        prt_3 = @(c,p,S,e) max(-S,min(0,c.*(p-e)));
        
        %New rule with scale up as a parameters
        pp_ps_pred_new = @(a,b,c,d,tpp,prt_elim1,prt_elim2,p,e,scale_ups) ...
            max(-S,d*(part_elim_calc(tpp,I,p,S,[prt_elim1 prt_elim2],scale_ups)) ... %)*(a<0.02 | a > 0.8)...
            + prt_2(d,b,p,S) + prt_3(c,p,S,e)+ prt_1(a,p)) ;%*(                 
            
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
            lb = [0    -1  -1   0 1     0    0 0   0   0 -1e4 0];%              1                0.01   0.01];
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
        
        interp_fit = auto_fit;
        
        err_int = fun(x_fit);
        err_o = fun(x_fit_opt);
%          figure(10); 
%         plot(prs,frs_real,'k','linewidth',2); hold on;
%         plot(prs,interp_fit,'r');
%         plot(prs,opt_fit,'g');
%         title(num2str(I));
end