# README.md

## Code Overview
<p>All files were developed on a MacBook Pro within MATLAB. The repository was last tested on a MacBook Pro running Ventura 13.4 using MATLAB_R2023a. <br>

<p>The code provided contains the simulations, prediction code and analyses used in “Pulsatile electrical stimulation disrupts neural firing in predictable, correctable ways” - Steinhardt, Mitchell, Cullen, Fridman. If using any of this code, please cite the paper.<br>

<p>Besides the code provided, the only requirements are built-in MATLAB optimization algorithms that are used to perform optimization of hyper parameter tuning in the prediction code.<br>

## System Requirements
<p>This code should run in a recent version of MATLAB on Mac, Windows, or Linux with the code and data in these packaged and the standard toolboxes, including Optimization Toolbox.<br>

## Installation Guide
Simply download the full repository and all functions should run with reference to data in the relevant_data folder. 

## Instructions for How to Use
The repository has three code folders which reference some helper_functions in the provided folder:
1) prediction_code 2) simulation_code 3) figure_making_files

<p>Functions are designed to be run within the directory they are located. With directory references (’..’) at the beginning of each file.<br>

**Predictions Code**
***
_PFR_fitting_demo.m_ - uses the same technique used in the paper to fit PFRS with pulse rates up to 350 pps at difference pulse amplitudes and spontaneous rates with smooth changing in hyperparameters. _fit_pr_fr_data.m_ performs the fitting to the left out data

  I: optimize fit from set starting points (example handful guesses)
Runtime about ~110 seconds - 1 spontaneous rate condition. With fitting and plotting/printing out error. 

  II: performs same fitting from left out cases where there was not hand fitting to initialize x0/hyperparameters.
    Interpolates between previous fits as starting parameterize for all pulse amplitudes. Runtime with no parpool ~555 seconds     with carpool 140 second - 1 spontaneous rate condition. Makes interpolated and interpolated and optimized fits <br>

<p>Based on the assumptions of the paper, code could be adapted to novel pulse rate and firing data for afferents or neurons. For best fitting, it is recommended to handfit a subset of hyperparameter at various pulse amplitudes then use the interpolation function with optimization from those best fits to find the functions that capture how pulse parameters and spontaneous activity of the afferent affect the pulse rate-firing rate relationship.<br>

**Simulation Code**
***
Code used to simulate vestibular afferents under a variety of pulsatile stimulation conditions and for afferents of different physiological properties. 

_simulation_demo.m_ gives a basic code for changing the pulse parameter (pulse rate and pulse amplitude) and running an experiment on an afferent with given regularity for a given length of time.<br>

If run_mode = “override,” use _set_override_v2.m_ - to change any basic stimulation parameters, neuron properties, or trial timing properties.<br>

*run_chosen_expt_3_3_22.m* is references, which includes built-in experiments used in a variety of previous work and this paper. <br>

<p>Experimental paradigms are changed by changing expt_num: <br>

  *expt_num = []  - case of any pulse parameter combination or DC mode (as used in previous paper) on particular afferent
Can use this mode to get spike and pulse times

  1. Uses same pulse rate parameters as in Mitchell at all but expects multiple test Is. (Override can change the pulse rate and pulse amplitude anyways and override this but save out blocked firing rates per same pulse rate and pulse amplitude condition.

  2. Same as 1 except set to use exact Mitchell experiment best match I-value for pulse block experiment with same pulse rate at Mitchell (2016) from Steinhardt, C. R., & Fridman, G. Y. (2020). Predicting response of spontaneously firing afferents to prosthetic pulsatile stimulation. In 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 2929-2933). IEEE.

  3. Designed to test any pulse rate and pulse amplitude parameter combination and run in parallel for speed

  4. Same as 3 but with setting for Mitchell parameters

  5. Pulse modulation experiments. Can be set for pulse amplitude modulation or pulse rate modulation. Uses the pulse_modulator.m to design the base pulse modulation waveform

  *6. Used to try any pulse rate and amplitude combination and save out data most useful for getting PFRS and related spike timing information for this paper. Plots PA v. FR and PR v. FR plots for multiple combinations

<p>Expts 1-4 were originally designed for Steinhardt & Fridman (2020). For any experiments not with reference to Mitchell et all (2016) expt.num = [], 3, 5, or 6 was used in this paper.<br>

The original afferent simulation code was developed in Steinhardt, C. R., & Fridman, G. Y. (2021). Direct current effects on afferent and hair cell to elicit natural firing patterns. Iscience, 24(3).
For details of how the afferent EPSC activity and channels are run, as found in *run_expt_on_axon_10_10_20_f.m, pulse_adapt_expt_indiv.m, make_axon_inputs_7_19_20.m*, and *synapse_adapt_5_28_20_f.m* please see the paper.

**Figure making files**
***
This code is labeled throughout with “Figure X X” with reference to figures in text and supplemental figures sections. Files use data from relevant_data folder throughout.

*main_fig_plotter.m*  
  I. uses a variety of saved runs under different conditions to produce traces with individual PFRs and fit
-under different spontaneous rate, jitter condition, varied firing regularity, low/medium/high I trace visualization, fitting comparisons, histogram of spike timing analysis.
  
  II. Demos a subset of the experiments described in I and contains code for visualizing outcome of these experiments.
  
  III. Code for visualizing the pulse trains, membrane potentials, and channel dynamics traces under different conditions and other codes used through Figures 1-5 unrelated to the slope analyses or the experimental data.

  *sim_slope_analysis.m* - uses the saved session runs to perform the slope and statistical analyses described in the test on   simulated data under described conditions<br>


*ephys_data_analysis_to_slope_n_metrics.m* 
  I. Takes the spike timing of recorded data in mitchell_data_all and transforms it into firing rates per block per simulation conditions. 

  II. Then this data is used to visualize PFRS and get slopes per spontaneous rate, pulse amplitude and pulse rate condition. 

  III. Permutation analysis and statistics are performed here against the simulation slope data produced in sim_slope_analysis.m 

  IV. Parameter visualization from the fits and rms error and statistics of the fits are performed in PFR_fitting_demo.m

  
**Relevant Data**
- Original simulation outputs in *vestib_sim_01_25_2023_allmu_10_reps*
- Mitchell experimenta data for all afferents in *mitchell_data_all*
- Compressed into per spontaneous rate format in the *pr_fr_S*01-15-2023.mat* files.
- Saved out fits and their rms information per all cases in *S_I_F_fits_2_24_23.mat*
- Hyperparameter info and starting points in *optimization_seed_params.mat*
- Runs under various conditions (low/medium/high I, irregular v. regular afferent, low/high conductances) in *specific_runs* folder. Referenced in main_fig_plotter.m
