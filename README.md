{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fswiss\fcharset0 Helvetica-Bold;\f2\fswiss\fcharset0 ArialMT;
\f3\fswiss\fcharset0 Arial-ItalicMT;\f4\fswiss\fcharset0 Arial-BoldMT;}
{\colortbl;\red255\green255\blue255;\red26\green26\blue26;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c13333\c13333\c13333;\cssrgb\c100000\c100000\c100000;}
\margl1440\margr1440\vieww13240\viewh12780\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # README.md\
\
## Code Overview\
All files were developed on a MacBook Pro within MATLAB. The repository was last tested on a MacBook Pro running Ventura 13.4 using MATLAB_R2023a. \
\
The code provided contains the simulations, prediction code and analyses used in \'93Pulsatile electrical stimulation disrupts neural firing in predictable, correctable ways\'94 - Steinhardt, Mitchell, Cullen, Fridman. If using any of this code, please cite the paper.\
\
Besides the code provided, the only requirements are built-in MATLAB optimization algorithms that are used to perform optimization of hyper parameter tuning in the prediction code.\
\
##System Requirements\
This code should run in a recent version of MATLAB on Mac, Windows, or Linux with the code and data in these packaged and the standard toolboxes, including Optimization Toolbox.\
\
##Installation Guide\
Simply download the full repository and all functions should run with reference to data in the relevant_data folder. \
\
##Instructions for How to Use\
The repository has three code folders which reference some helper_functions in the provided folder:\
1) prediction_code 2) simulation_code 3) figure_making_files\
\
Functions are designed to be run within the directory they are located. With directory references (\'92..\'92) at the beginning of each file.\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \ul \ulc0 Predictions Code\
***\ulnone \

\f1\b PFR_fitting_demo.m
\f0\b0  - uses the same technique used in the paper to fit PFRS with pulse rates up to 350 pps at difference pulse amplitudes and spontaneous rates with smooth changing in hyperparameters. 
\f1\b fit_pr_fr_data.m
\f0\b0  performs the fitting to the left out data\
\
I: optimize fit from set starting points (example handful guesses)\
Runtime about ~110 seconds - 1 spontaneous rate condition. With fitting and plotting/printing out error. \
\
II: performs same fitting from left out cases where there was not hand fitting to initialize x0 /hyperparameters.\
Interpolates between previous fits as starting parameterize for all pulse amplitudes. Runtime with no parpool ~555 seconds with carpool 140 second - 1 spontaneous rate condition.\
Makes interpolated and interpolated and optimized fits\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \
Based on the assumptions of the paper, code could be adapted to novel pulse rate and firing data for afferents or neurons. For best fitting, it is recommended to handfit a subset of hyperparameter at various pulse amplitudes then use the interpolation function with optimization from those best fits to find the functions that capture how pulse parameters and spontaneous activity of the afferent affect the pulse rate-firing rate relationship.\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \ul \ulc0 Simulation Code\
***\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \ulnone Code used to simulate vestibular afferents under a variety of pulsatile stimulation conditions and for afferents of different physiological properties. \
\

\f1\b simulation_demo.m
\f0\b0  gives a basic code for changing the pulse parameter (pulse rate and pulse amplitude) and running an experiment on an afferent with given regularity for a given length of time.\
\
If run_mode = \'93override,\'94 use 
\f1\b set_override_v2.m
\f0\b0  - to change any basic stimulation parameters, neuron properties, or trial timing properties.\
\

\f1\b run_chosen_expt_3_3_22.m 
\f0\b0 is references, which includes built-in experiments used in a variety of previous work and this paper. Experimental paradigms are changed by changing expt_num:\
\
*expt_num = []  - case of any pulse parameter combination or DC mode (as used in previous paper) on particular afferent\
Can use this mode to get spike and pulse times\
1: Uses same pulse rate parameters as in Mitchell at all but expects multiple test Is. (Override can change the pulse rate and pulse amplitude anyways and override this but save out blocked firing rates per same pulse rate and pulse amplitude condition.\
2: Same as 1 except set to use exact Mitchell experiment best match I-value for pulse block experiment with same pulse rate at Mitchell (2016) from 
\f2\fs26 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Steinhardt, C. R., & Fridman, G. Y. (2020). Predicting response of spontaneously firing afferents to prosthetic pulsatile stimulation. In\'a0
\f3\i 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)
\f2\i0 \'a0(pp. 2929-2933). IEEE.
\f0\fs24 \cf0 \cb1 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 \
*3: Designed to test any pulse rate and pulse amplitude parameter combination and run in parallel for speed\
4: Same as 3 but with setting for Mitchell parameters\
5: Pulse modulation experiments. Can be set for pulse amplitude modulation or pulse rate modulation. Uses the 
\f1\b pulse_modulator.m 
\f0\b0 to design the base pulse modulation waveform\
*6: Used to try any pulse rate and amplitude combination and save out data most useful for getting PFRS and related spike timing information for this paper. Plots PA v. FR and PR v. FR plots for multiple combinations\
pulse_adapt_gen - blocks of fixed pulse rate and pulse amplitude stimulation as chosen in advance, adaptation effect\
pulse_adapt_best_mitchell - specific to the Mitchell parameters\
\
Expts 1-4 were originally designed for Steinhardt & Fridman (2020). For any experiments not with reference to Mitchell et all (2016) expt.num = [], 3, 5, or 6 was used in this paper.\
\
The original afferent simulation code was developed in 
\f2\fs26 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Steinhardt, C. R., & Fridman, G. Y. (2021). Direct current effects on afferent and hair cell to elicit natural firing patterns.\'a0
\f3\i Iscience
\f2\i0 ,\'a0
\f3\i 24
\f2\i0 (3).\
For details of how the afferent EPSC activity and channels are run, as found in 
\f4\b run_expt_on_axon_10_10_20_f.m, pulse_adapt_expt_indiv.m
\f2\b0 , 
\f4\b make_axon_inputs_7_19_20.m, 
\f2\b0 and 
\f4\b synapse_adapt_5_28_20_f.m 
\f2\b0 please see the paper.
\f0\fs24 \cf0 \cb1 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 \
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \ul \ulc0 Figure making files \
***\
\ulnone This code is labeled throughout with \'93Figure X X\'94 with reference to figures in text and supplemental figures sections. Files use data from relevant_data folder throughout.\

\f1\b \
main_fig_plotter.m 
\f0\b0 - \
I. uses a variety of saved runs under different conditions to produce traces with individual PFRs and fit\
-under different spontaneous rate, jitter condition, varied firing regularity, low/medium/high I trace visualization, fitting comparisons, histogram of spike timing analysis.\
II. Demos a subset of the experiments described in I and contains code for visualizing outcome of these experiments.\
III. Code for visualizing the pulse trains, membrane potentials, and channel dynamics traces under different conditions and other codes used through Figures 1-5 unrelated to the slope analyses or the experimental data.\

\f1\b \
sim_slope_analysis.m
\f0\b0  - uses the saved session runs to perform the slope and statistical analyses described in the test on simulated data under described conditions\

\f1\b \
ephys_data_analysis_to_slope_n_metrics.m
\f0\b0  - \
I. Takes the spike timing of recorded data in mitchell_data_all and transforms it into firing rates per block per simulation conditions. \
II. Then this data is used to visualize PFRS and get slopes per spontaneous rate, pulse amplitude and pulse rate condition. III. Permutation analysis and statistics are performed here against the simulation slope data produced in sim_slope_analysis.m \ul \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \ulnone IV.Parameter visualization from the fits and rms error and statistics of the fits are performed in PFR_fitting_demo.m\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \ul \ulc0 Relevant Data\ulnone \
Original simulation outputs in vestib_sim_01_25_2023_allmu_10_reps\
Mitchell data in mitchell_data_all\
Compressed into per spontaneous rate format in the pr_fr_S*01-15-2023.mat files.\
Saved out fits and their rms information per all cases in S_I_F_fits_2_24_23.mat\
Hyperparameter info and starting points in optimization_seed_params.mat\
Runs under various conditions (low/medium/high I, irregular v. regular afferent, low/high conductances) in specific_runs folder. Referenced in main_fig_plotter.m\
\
\
}