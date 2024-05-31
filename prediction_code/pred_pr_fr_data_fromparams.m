function [err_int,fin_pred] = pred_pr_fr_data_fromparams(S,I,prs,frs_real,x_fit)
%Based on fit_pr_fr_data.m - see for full implementation with optimization:
%Basic function for apply pulsatile stimulation rules to untested pulse
%parameter and neuron conditions
%Outputs:
%err_int (prediction error from real),fin_pred (PFR prediction)
%Inputs: S(spont rate), I(pulse amplitude), prs (pulse rate), frs_real,
%(real responses for given parameters)
%x_fit (parameters for fitting chosen in some way)

% Copyright 2023 Cynthia Steinhardt
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_fit_opt = x_fit;

%Fit Equations:
prt_1 = @(a,p) min(a.*p,p);
prt_2 = @(d,b,p,S) max(-d*S,d*b.*p);
prt_3 = @(c,p,S,e) max(-S,min(0,c.*(p-e)));

%Fit Equations:
prt_1 = @(a,p) min(a.*p,p);
prt_2 = @(d,b,p,S) max(-d*S,d*b.*p);
prt_3 = @(c,p,S,e) max(-S,min(0,c.*(p-e)));

%New rule with scale up as a parameters
pp_ps_pred_new = @(a,b,c,d,tpp,prt_elim1,prt_elim2,p,e,scale_ups) ...
    max(-S,d*(part_elim_calc(tpp,I,p,S,[prt_elim1 prt_elim2],scale_ups)) ... %)*(a<0.02 | a > 0.8)...
    + prt_2(d,b,p,S) + prt_3(c,p,S,e)+ prt_1(a,p)) ;%*(

if (S==0)

    fac_fun = @(p,p_cut,sig_scale) 1./(1+exp(sig_scale.*(-p + p_cut))); % p= prs here

    %Standard equations including the facil part
    error_fun_handfit = @(a,b,c,d,t_pp,prt_elim1,prt_elim2,frs_pred,p,e,scale_up1,scale_up2,p_facil,sig_scale) ...
        mse(frs_pred -  fac_fun(prs,p_facil,sig_scale).*pp_ps_pred_new(a,b,c,d,t_pp,prt_elim1,...
        prt_elim2,p,e,[scale_up1 scale_up2])); %error function

    fun = @(x) error_fun_handfit(x(1),x(2),x(3),x(4),x(5),x(6),x(7),...
        frs_real,prs,x(8),x(9),x(10),x_fit(11),x_fit(12));

    auto_fit = fac_fun(prs,x_fit_opt(11),x_fit_opt(12)).*...
        pp_ps_pred_new(x_fit_opt(1),x_fit_opt(2),x_fit_opt(3),x_fit_opt(4),...
        x_fit_opt(5),x_fit_opt(6),x_fit_opt(7),prs,x_fit_opt(8),[x_fit_opt(9) x_fit_opt(10)]);

else

    error_fun_handfit = @(a,b,c,d,t_pp,prt_elim1,prt_elim2,frs_pred,p,e,scale_up1,scale_up2) ...
        mse(frs_pred - pp_ps_pred_new(a,b,c,d,t_pp,prt_elim1,prt_elim2,p,e,[scale_up1 scale_up2])); %error function

    fun = @(x) error_fun_handfit(x(1),x(2),x(3),x(4),x(5),x(6),x(7),...
        frs_real,prs,x(8),x(9),x(10));%perS.t(1).pr_fr(n,:)-perS.t(1).pr_fr(1,:))


    auto_fit = pp_ps_pred_new(x_fit_opt(1),x_fit_opt(2),x_fit_opt(3),x_fit_opt(4),...
        x_fit_opt(5),x_fit_opt(6),x_fit_opt(7),prs,x_fit_opt(8),[x_fit_opt(9) x_fit_opt(10)]);

end

interp_fit = auto_fit;

fin_pred = auto_fit;%auto_fit;
err_int = fun(x_fit);
%          figure(10);
%         plot(prs,frs_real,'k','linewidth',2); hold on;
%         plot(prs,fin_pred,'r');
%         title(num2str(I));
end