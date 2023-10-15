%% Fpp Calculation Function
%Needs editting to do the part elim part well

%TWO FORMER SCALING RULES:
%scale_up = 5*(((I+dynl_scale)/300).^(4));
%scale_fact = 200/z_drop_scale;%/max(1,S); % replace with scale up for bend
%1 and 2, etc.
function [corr_ipi] = part_elim_calc(t_block,I,prs,S,prt_elims_tmp,scale_ups)
clear perc_elim_tmp

I_zdrop = 190;

%How many bends to do part elim at:
num_bends = max(3,ceil(t_block/(1e3./600)));
bend_nums = ceil(t_block./(1e3./prs));
if (I > I_zdrop)
    num_bends = 3;
end
%For each bend do the partial elim
for n_bend = 1:(num_bends-1)
    
    if n_bend == 1
        scale_up = scale_ups(1);
    else
        scale_up = scale_ups(2);
    end
    t_pp_full12 = t_block;%first bend only - block window
    
    %Switch point for each bend
    ipi_switch = t_block/n_bend;%ms
    pr_full_elim = (1e3./ipi_switch);%pps -for a given bend
    prs(1)= 1;
    %%%Partial elimination zone:
    if n_bend == 1
        %%%For bend one
        part_elim_frac = prt_elims_tmp(1);
    else
        %For bends 2, 3,.... using same number
        part_elim_frac = prt_elims_tmp(2);
    end
    
    %For any bend:
    pr_prt_elim = pr_full_elim -  min(.99,part_elim_frac).*(1e3./t_block);
    %sim_data(1).key_pts(n_cur,(n_bend-1)*2+2+1);
    t_pp_prt = 1e3/pr_prt_elim;
    t_pp_full = 1e3/(pr_full_elim); %ms
    
    %range for all bends if not kill off case:
    restrict_rng = (((1e3./prs - t_pp_full) >= 0) & ((t_pp_prt - 1e3./prs ) >= 0));

%     f_new(n_bend,:) = min(1,(1+scale_up)*(1 - (((1e3./prs) - t_pp_full)./(t_pp_prt - t_pp_full)))).*...
%                     restrict_rng;  
% %     
    %Linear version (newest): 
    lin_fr =@(t_pp_full,t_pp_prt,prs) ((1-((1e3./t_pp_full - prs)/(1e3/t_pp_full - 1e3/t_pp_prt)))).*restrict_rng;
 
    %orig per bend
    exp_fr =@(t_pp_full,t_pp_prt,prs,scale_up) min(1,(1+scale_up)*(1 - (((1e3./prs) - t_pp_full)./(t_pp_prt - t_pp_full)))).*...
                    restrict_rng; 
    %Bend 2: High I
    %if n_bend == 2
    full_k_fr=@(t_pp_full,t_pp_prt,prs,scale_up)  scale_up*((1-(((1e3./prs) - t_pp_full)/(t_pp_prt - t_pp_full))).^(3).*((t_pp_prt - 1e3./prs) >= 0));

    %if n_bend ==1
    dyn_l_fr =@(t_pp_full,t_pp_prt,prs,scale_up) ceil(scale_up*((((1e3./prs) - (t_pp_full-.001))/(t_pp_prt - t_pp_full)))).*restrict_rng;
   
    if (I > I_zdrop) % Where get suppression!
      if n_bend == 1
        if (S <= 4) &(I > I_zdrop+10) % only time seeing dynamic loops?
            perc_elim_tmp(n_bend,:)= dyn_l_fr(t_pp_full,t_pp_prt,prs,scale_up);
            %exp_fr(t_pp_full,t_pp_prt,prs,scale_up) + ...
            %    max(0,(dyn_l_fr(t_pp_full,t_pp_prt,prs,scale_up)-1));
        else %Normal rule
            perc_elim_tmp(n_bend,:) = exp_fr(t_pp_full,t_pp_prt,prs,scale_up);
        end
        
      else
          %Full suppression
          perc_elim_tmp(n_bend,:) = full_k_fr(t_pp_full,t_pp_prt,prs,scale_up);
      end
            
    else
       perc_elim_tmp(n_bend,:) = exp_fr(t_pp_full,t_pp_prt,prs,scale_up);
    end
 
end
%Get part and full elim part of bend
perc_elim = sum(perc_elim_tmp,1);%

if (I > I_zdrop)
    corr_ipi = (prs./(min(ceil(t_pp_full12./((1e3./prs))),2) + perc_elim));%.*facill_eq;%.*(sol_temp).*(sol_temp >= .75);
    
else
    corr_ipi = (prs./(ceil(t_pp_full12./((1e3./prs))) + perc_elim));%.*facill_eq;%.*(sol_temp).*(sol_temp >= .75);
end
corr_ipi(1) = 0;

end

% %% NOTES:
%     %%  Plot different functions against each other:       %  
%     scale_ups  = [.01 .1 1 5 10];
%     for n_s = 1:length(scale_ups)
%         scale_up = scale_ups(n_s)
%     figure(n_s);
%     subplot(2,1,1);
%   %  plot(prs,lin_fr(t_pp_full,t_pp_prt,prs),'k'); hold on;
%      plot(prs,exp_fr(t_pp_full,t_pp_prt,prs,scale_up),'b');
%  %    plot(prs,full_k_fr(t_pp_full,t_pp_prt,prs,scale_up),'r');
%  %    plot(prs,dyn_l_fr(t_pp_full,t_pp_prt,prs,scale_up),'g');
%      title(num2str(scale_up));
%      subplot(2,1,2);
%      
%  %   plot(prs,prs./(ceil(t_pp_full./((1e3./prs))) + lin_fr(t_pp_full,t_pp_prt,prs)),'k'); hold on;
%      plot(prs,prs./(ceil(t_pp_full./((1e3./prs)))+exp_fr(t_pp_full,t_pp_prt,prs,scale_up)),'b');
%  %    plot(prs,prs./(ceil(t_pp_full./((1e3./prs)))+full_k_fr(t_pp_full,t_pp_prt,prs,scale_up)),'r');
%      plot(prs,prs./(ceil(t_pp_full./((1e3./prs)))+dyn_l_fr(t_pp_full,t_pp_prt,prs,scale_up)),'g');
%     end

%% OLD WAY:
   %%
% %     %%%%%%%%%%%%% (Think about how to simplify)
% %     %Drop features:
%TWO DIFFERENT SCALINGS GOING TO JUST HAVE AS A PARAMETER FOR NOW:
% %     scale_up = 5*(((I+dynl_scale)/300).^(4));
% %     if (I > I_zdrop) %I = 200 for sim with no spont firing
% %         %Rule that does partial elimination where the tPP builds and
% %         %so partial elimination becomes full elimination and 2/3
% %         %becomes stronger
% %         
% %         if n_bend == 2
% %             % Full kill is like deep partial elimination:
% %             scale_fact = 200/z_drop_scale;%/max(1,S);
% %             perc_elim_tmp(n_bend,:) = scale_fact*((1-(((1e3./prs) - t_pp_full)/(t_pp_prt - t_pp_full))).^(3).*((t_pp_prt - 1e3./prs) >= 0));
% %             
% %         else
% %             
% %             if  (S<4)
% %                 %   First dip is dynamics getting trapped in a loop modeled
% %                 %  like this (b/c pulses get looped with selves):
% %                 tmp_perc = ((((1e3./prs) - t_pp_full)/(t_pp_prt - t_pp_full))).*...
% %                     (((1e3./prs - t_pp_full) > 0) & ((t_pp_prt - 1e3./prs ) >= 0));
% %                 perc_elim_tmp(n_bend,:) = ceil(tmp_perc*scale_up);%round(tmp_perc*scale_up);%scales up when amplitude really high
% %            
% %             else
% %                 
% %                 perc_elim_tmp(n_bend,:) = min(1,(1+scale_up)*(1 - (((1e3./prs) - t_pp_full)./(t_pp_prt - t_pp_full)))).*...
% %                     (((1e3./prs - t_pp_full) >= 0) & ((t_pp_prt - 1e3./prs ) >= 0));               
% %   
% %             end
% %         end
% %         
% %     else
   
% %         perc_elim_tmp(n_bend,:) = min(1,(1+scale_up)*(1 - (((1e3./prs) - t_pp_full)./(t_pp_prt - t_pp_full)))).*...
% %                     (((1e3./prs - t_pp_full) >= 0) & ((t_pp_prt - 1e3./prs) >= 0));
% %     end