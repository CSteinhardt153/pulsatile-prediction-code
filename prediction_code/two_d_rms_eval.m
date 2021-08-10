function [I_idx,rms_best, fr_pred_best] = two_d_rms_eval(S_cur,prs_cur,real_y)
%Function for 2d rms  between full prediction and the input data
%%%%%%%%%%%%%%%%
if (size(real_y,1) < size(real_y,2))
    real_y = real_y';
end
prs_tot = 0:400;
I_range = [0:5:500]%360];%[0:360];
for n_Is = 1:length(I_range)
    I_cur = I_range(n_Is);
     [tot_pred ] = interp_pred_f_5_5_21(I_cur,S_cur,prs_tot);
    %[tot_pred] = interp_pred_fr_v2(I_cur,S_cur,prs_tot);
    
    fr_preds(n_Is,:) = S_cur  + tot_pred';
end

for n_prs = 1:length(prs_cur)
    
    real_x = prs_cur(n_prs);
    
    x_diff = prs_tot - real_x;
    x_diff_repped = repmat(x_diff,[length(I_range) 1]);
    y_diff = fr_preds - real_y(n_prs,1);
    dist_y  =(y_diff).^2;
    %Correct off std in y/fr
    %dist_y  =max(0,dist_y - real_y(n_prs,2)^2);
    q =  sqrt((x_diff_repped).^2 +  dist_y);%+ (y_diff).^2);
    %Subtract out anything within std

    [rms_val_pt_n I_idx] = min(q');%min val per current amplitude
    rms_per_pt(n_prs,:) = rms_val_pt_n;
    
end
shape_err = sum(rms_per_pt,1); %total error per current from best point


% comp_preds = fr_preds(:,find(ismember(prs_tot,prs_cur)));
% for n_preds = 1:size(comp_preds,1)
% [pred_corrs(n_preds,:) lags] = xcorr(real_y',comp_preds(n_preds,:));
% end
%
% normed_xcorr_0 = pred_corrs(:,find(lags == 0));
% normed_xcorr_0 = (1 - normed_xcorr_0./max(normed_xcorr_0));

%= cross_pt_rms;%2*(cross_pt_rms/max(cross_pt_rms));%normed_xcorr_0' +

[rmss idx_fins] = sort(shape_err);

tot_err = rmss + rmss.*.5.*idx_fins./200;%shape_err + I_idx;
%[rms_tmp idx_tmp] = sort(tot_err );
[rms_2 idx_2] = min(tot_err );
I_idx = I_range(idx_fins(idx_2));
fr_pred_best = fr_preds(idx_fins(idx_2),:);

rms_best = rmss(idx_2);%(1);
% % cols = jet(length(75:98))
% % figure(100); 
% % for n= 75:98
% % plot(prs_tot,fr_preds(n,:),'color',cols(n-74,:)); hold on;
% % end
% % errorbar(prs_cur,real_y(:,1),real_y(:,2),'ko'); hold on;
% %  plot(prs_tot,fr_pred_best,'k','linewidth',2); 
end