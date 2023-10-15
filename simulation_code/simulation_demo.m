%simulation_demo.m
%Code that shows the main builting experiments in the simulation code and
%how to use it with adapted parameters for neuronal physiology, stimulation
%parameters, and stimulation paradigm
%Code was originally developed and used in:
%-Steinhardt, C. R., & Fridman, G. Y. (2021). Direct current effects on afferent and hair cell to elicit natural firing patterns. Iscience, 24(3).
%More details in the paper above.
%Last Updated 10.14.23 CRS
%=========================================================================
expt.num = []; %could be [[],1 - 6]
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
pulse_rates = [25:25:300]; %pps
I_targ = 80;%uA
I_norm = -20; %normalization factor
use_curs = I_targ/I_norm;
[override] = set_overrides_v2(run_mode,output,{'pulse_rate',pulse_rates},{'curr_options',use_curs},{'tot_reps', 1},{'mu_IPT',1},{'sim_time',1150});

%Example code runs experiment at pulse rates 25 to 300 at I = 48 uA for
%1150 ms with a neuron with mu = 1 (low spontaneous firing rate ~30) in a
%visualization mode whre the EPSCS, pulse timing, trace and channel
%dynamics can be visualized. All these setting can be modified in
%set_overrides_v2 or in lines 37-41

%RUN EXPERIMENTS FROM HERE ('exact') settings are in this code:
run_chosen_expt(expt,run_mode,override,output);
