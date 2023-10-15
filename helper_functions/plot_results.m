function plot_results(perS,n_t,line_color,plot_num)
%Fitting of first pp_ps_pred
frs_pred_fin  =perS.t.pr_fr;
prs = perS.prs;
S = perS.S;
Fpp = perS.Fpp;
fin_vars = perS.t(n_t).fin_vars(2:end,:);
Is = perS.I;
frs_pred_fin = perS.t(n_t).pr_fr;
pp_ps_pred = @(a,b,c,d,e,fpps) d*fpps  + min(a.*prs,prs)*(d ~= 1 | e ~= 1) + ...
    (max(-S,b.*prs)*(d == 1 | e == 1) + max(-S,c.*prs))*(a==0) ;% add sparsity regularizer if doesn't go well?

for n = 1:44
    figure(plot_num); subplot(5,9,n)
    plot(prs,frs_pred_fin(n,:) - frs_pred_fin(1,:),'k'); hold on;
    plot(prs,pp_ps_pred(fin_vars(n,1),fin_vars(n,2),fin_vars(n,3),...
        fin_vars(n,4),fin_vars(n,5),Fpp(n,:)),'color',line_color);
    title(num2str(Is(n)))
    %%%Breakdown of contributing variables
    %  subplot(2,2,1:2);
    % plot(prs,frs_pred_fin(n,:) - frs_pred_fin(1,:),'k'); hold on;
    % %plot(prs,Fpp(n,:),'b');
    % plot(prs,pp_ps_pred(fin_vars(n,1),fin_vars(n,2),fin_vars(n,3),fin_vars(n,4),fin_es(n),Fpp(n,:)),'r');
    % title(num2str(Is(n)))
    % subplot(2,2,3);
    % plot(prs,max(-S,min(fin_vars(n,2).*prs,prs)),'g'); hold on;
    % plot(prs,max(-S,min(fin_vars(n,3).*prs,prs)),'m');
    % legend('p_{pxs}','p_{sxp}'); title ('Blocking effects')
    % subplot(2,2,4);
    % plot(prs,max(-S,min(fin_vars(n,1).*prs,prs)),'g'); hold on;
    % plot(prs,fin_vars(n,4).*ones(size(prs)),'m');
    % legend('Facil','P_{pAP}')
    % pause(2); close all;
end
end
