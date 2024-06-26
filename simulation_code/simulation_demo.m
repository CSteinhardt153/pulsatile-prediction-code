%simulation_demo.m
%Code that shows the main builting experiments in the simulation code and
%how to use it with adapted parameters for neuronal physiology, stimulation
%parameters, and stimulation paradigm
%Code was originally developed and used in:
%-Steinhardt, C. R., & Fridman, G. Y. (2021). Direct current effects on afferent and hair cell to elicit natural firing patterns. Iscience, 24(3).
%More details in the paper above.
%Last Updated 10.14.23 CRS
%
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
%=========================================================================
expt.num = [];%[]; %could be [[],1 - 6]
run_mode = 'override';%  {'exact','override'}; %can run the exact experiment from study, override some parameters, or do a new experiment
%%% If choose override skip to line 39 to edit otherwise select experiment

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Deciding what outputs to visualize:
output.vis_plots = 0; %If want to see the afferent model APs without experiments
output.vis_plot_num = 6; %plot number when visualizing
output.label_aps = 0; %This is meaningless if don't set the seed, etc. - for check if AP spont or pulse induced
output.all_spk = 0;
output.do_phase_plane_anal = 0;
output.demo_pulse_mod = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Experiment settings:
%%%%%%%%%%%%%% Different experiments
%Can choose to run all three experiments or just one:

%expt.num = [] 1-6 information about them in the readme. Setting [], 3, 5 and
%6 most useful for this paper. expt.num = [] gives a single run
%visualization. 3, 5, and 6 are used for running multiple pulse parameter
%conditions at once. 5 is for PRM or PAM. Experiments 1,2, and 4 are for
%comparison to Mitchell (2016) experiments.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% If overriding - choosing specific parameters etc:
pulse_rates = 4;%[50 150 250];%1;%[25:25:300]; %pps
I_targ = 20;%uA
I_norm = -20; %normalization factor
use_curs = I_targ/I_norm;
[override] = set_overrides_v2(run_mode,output,{'pulse_rate',pulse_rates},...
    {'curr_options',use_curs},{'tot_reps', 1},{'mu_IPT',.5},{'sim_time',1150});

%Example code runs experiment at pulse rates 25 to 300 at I = 48 uA for
%1150 ms with a neuron with mu = 1 (low spontaneous firing rate ~30) in a
%visualization mode whre the EPSCS, pulse timing, trace and channel
%dynamics can be visualized. All these setting can be modified in
%set_overrides_v2 or in lines 37-41

%RUN EXPERIMENTS FROM HERE ('exact') settings are in this code:
run_chosen_expt(expt,run_mode,override,output);
