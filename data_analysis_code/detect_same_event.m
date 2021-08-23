function [event_idxs ] = detect_same_event(event_1,event_2,timing,thresh,min_inter_event_time)
%For detecting same pulse or spike, etc. or if spike/pulse within same window.
%If detecting if same event make event_1= event_2;
%If checking if event occurs within time of other event (e.g. "spike" within time of stimulation)
%make event1 the voltage and event 2 the pulse train.
%Comparing two found events (Case 2) then both should be in indices
%only identifies same event as the first time point that agrees with rules.
%(Here, a threshold being crossed and first cross). Eliminate repeated
%counts of same event.
%Min inter_event_time needs to be in same units as timing or in intervals
%of the timing vector for correct calculation
%Started 8/2/21
%Last Updated 8/2/21
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_it = 0;

%Find events in raw trace using thresholds and time windows
if (length(event_1) == length(event_2))
    
    x_thresh_times = (event_1 > thresh);
    corr_event_times = find(x_thresh_times);
    if(sum(event_1 == event_2) == length(event_1))
        %Check if in time of event_2
        not_too_close = (diff(corr_event_times) > min_inter_event_time);
    else
        print('Issue - add more catches')
    end
    
    event_idxs = corr_event_times(not_too_close)-1;
    if plot_it
        figure(2);
        plot(timing,event_1,'k'); hold on;
        plot(timing(event_idxs),event_1(event_idxs),'r.');
    end
else
    
    for n_ts = 1:length(event_1)
        too_close(n_ts) = sum(((event_1(n_ts) - event_2) < min_inter_event_time) & ...
            ((event_1(n_ts) - event_2) > 0));
    end
    min_inter_in_time = min_inter_event_time* min(diff(timing)) % for pulses would want to be ~ 2 ms
    event_idxs = event_1(~too_close) ;
end


end

