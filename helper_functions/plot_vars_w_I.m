%% Result plotting functions
%  Best fit as I increases
function plot_vars_w_I(perS,n_t,line_color,plot_num)
fin_vars= perS.t(n_t).fin_vars(2:end,1:4);
fin_es= perS.t(n_t).fin_vars(2:end,5);
Is = perS.I(2:end);

%fin_vars = fin_a_b_c_d;%(2:end,:);
%fin_es = fin_e;%(2:end);
figure(plot_num);subplot(5,1,1); plot(Is,fin_vars(:,1),'color',line_color);
hold on;
ylabel('P_{facil}')
subplot(5,1,2); plot(Is,fin_vars(:,2),'color',line_color);hold on;ylabel('P_{pxs}')
subplot(5,1,3); plot(Is,fin_vars(:,3),'color',line_color);hold on;ylabel('P_{sxp}')
subplot(5,1,4); plot(Is,fin_vars(:,4),'color',line_color);hold on;ylabel('P_{pAP}')
subplot(5,1,5); plot(Is,fin_es,'color',line_color);hold on;ylabel('Have pulses reached 1:1')
end