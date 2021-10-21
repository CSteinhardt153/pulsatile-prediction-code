# pulstaileDir

## Overview
_simulation_code_
- fin_pp_sp_fig_plotter.m plots runs all the main experiments used in the paper
- pulse_XX_XX.m are the names of the files that support different experiments (e.g. pulse_rate_modulator.m is for PRM/PAM)
- set_overrides_v2.m - where can choose experimental parameters versus run a standard experiment if no extra variables are added

_prediction_code_
- ephys_data_analysis_5_9_21.m is the main file that imports data and run the prediction algorithm and data analysis
- - currently works with data processed by DM

_data_analysis_code_
- made pp_sp_sort.m to sort spont v. pulse artifact time to get more accurate analysis of pulse rate firing rate relationship
