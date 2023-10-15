function [override] = set_overrides_v2(run_mode, output,varargin)
%Allows general simulation setting to be overridden
%simulation length, pulsatile/dc
%Neuron regularity, conductances, epsc rates, adaptation of hair cell to
%current (should not occur with pulsatile stimulation)
%Packages for overriding standard settings in 'exact' simulation of
%experiments from Mitchell, Della Santina, Cullen (2016) paper
%Add varargin to change specific variables in format {variable name, value}
%(e.g. [override] = set_overrides_v2('override',
%output,{'pulse_rate',2},{'curr_options',[-3 -6]},{tot_reps, 4});)
%Started 12/28/20
%Last Updated 5/2/21 CRS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(run_mode,'override')
   %EITHER DO BY HAND HERE: 
%%%%%%%%%%%%%%%%MAIN EXPERIMENTAL PARAMETERS
pulse_rate = [40 90 180]; %linspace(0,300,31);%[125 150 175 200]; 
curr_options = -3;%linspace(0,-18,51);

%%%%%%%%%%%%%%%%%SIMULATION SETTINGS
sim_info.curr_scaling = 1;
sim_info.tot_reps = [1];% Number of repetitions fo experiment
sim_info.sim_time = 1150; %ms Simulation length
sim_info.sim_start_time = 150; %start stimulate part of experiment (usually 1 ms)
sim_info.sim_end_time = [];
sim_info.dt = 1e3; %1/1000 of a ms
sim_info.inj_cur = [1 1];%Allow posibility of (1) injected current and (2) epsp release
sim_info.isDC = 0;
sim_info.non_quant = 1;
sim_info.low_cond = [0]; % Axon is low cond like Manca expt or high cond like Goldberg
sim_info.isPlan = 0;
sim_info.do_jitter = 0;% jitter the pulse timing to test if effects hold true
sim_info.full_seq = 0;%control the full sequence of pulses to be specific on the exact shape of a pulse
%Can manipulate here to try diff scenarios. If empty - chooses by what expt
%was performed on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Only transient/type I/irreg modeled here
%Can be overridden in experiment section:

%For incorportating adaptation function: (DC)
sim_info.do_adapt = 0; %do an adaptation in mu at all (for DC)
sim_info.do_orig = 0; %do it in the original adaptation way of the paper v. the modified way

%%%%%%%%%%%%%%%%%%Adaptation function settings (DC/hair cell):
sim_info.g_o_1 = 3.25*.15; %low cond model
sim_info.g_o_2 = 3.25*.9;
% What the taus should be?
sim_info.tau_1 = 2;%Manca 2.26, final 2
sim_info.tau_2 = .3;%Manca 0.24, final .3
sim_info.sense_scale =  0;%0.25;
sim_info.fr_o = 45;

sim_info.is_reg = 0;
sim_info.doIh = 0; %hyperpolarization-activated cationic current
%Simulation parameters generically
sim_info.epsc_sample= .1;
sim_info.epsc_over_ride = 0;%choose exact EPSC timing/height ( for tref experiment)


%%%%%%%%%%%%%%%%%%% Neuron type -conductance, K, mu settings for an  irregular
range_scale = 1;
if ~sim_info.is_reg 
%Irregular:
sim_info.mu_IPT = .25;%[.25 .5 1 2]
sim_info.epsc_scale = 1;
else
%Regular
sim_info.mu_IPT = 4;
sim_info.epsc_scale = .04;
end

%%%set sensitivity and adaptation measures
sim_info.scale_adapt_mu = 0;

%MODEL from last EMBC paper
gNas = range_scale*13;%6*13 (these are for normal axon model). The used are for mitchell (lower firing model)
gKHs = range_scale*2.8;%4*2.8;

if ~sim_info.is_reg 
%Irregular
gKLs = 1;%.75;
else
%Regular
gKLs = 0;%.75;
end

   %OR WITH VARARGIN HERE: (Will make the final changes from above) 
if ~isempty(varargin) % If want to just choose a few variables and use the other presets or override settings above
for n_ins = 1:length(varargin)
    change_n = varargin{n_ins};
    %Make sure is a real field
    if isfield(sim_info,change_n{1})
        %Currently all inputs are numbers so this format should work
        eval(['sim_info.' change_n{1} ' = [' num2str(change_n{2}) '];'])
        
        %If statement to prevent overriding an override
        if (strcmp(change_n{1},'epsc_scale')) |(strcmp(change_n{1},'gKLs'))
        else
        if ~sim_info.is_reg
            %Irregular
            gKLs = 1;%.75;
            sim_info.epsc_scale = 1;
        else
            %Regular
            gKLs = 0;%.75;
            sim_info.epsc_scale = .025;%.2;%.15;%.004;
        end
        end

    else
        if exist(change_n{1},'var')
            eval([change_n{1} ' = [' num2str(change_n{2}) '];']);% just incase is vector
        else
        warning([change_n{1} ' is not an input variable or a field in sim_info.']);
        end
    end
    
end

else
end

%Model from DC paper
if sim_info.isDC
gNas = 6*13;
gKHs = 4*2.8;
end
%%%%%%%%%%%%%Package variables for studies
override.sim_info = sim_info;
override.gNas= gNas;
override.gKHs= gKHs;
override.gKLs= gKLs;
override.pulse_rate = pulse_rate;
override.curr_options = curr_options;
override.output = output;
else
    override = []; % Use the original settings
end
end

