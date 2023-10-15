function [z_drop_scale, dynl_scale,t_pps,prt_elim_perc] = get_rel_hand_fit(S_cur,I_cur)
load('pulse_rule_handfit_vars_1_6_22.mat','I_in_t_block','prt_elim_S','zero_drop_scale','dynamic_loop_scale','Ss')


%Find relevant column :

     x_range_tblck(1).range = min(I_in_t_block (:,1)):.1:max(I_in_t_block (:,1));
            x_range_tblck(2).range =  min(I_in_t_block (:,2)):.1:max(I_in_t_block (:,2));
            %  [pred_t_block] = interp_SI_funs(I_in_t_block (:,1),I_in_t_block (:,2),x_range_tblck, I_cur,0); % Cannot predict outside of range of x input..
            %t_b
            [t_pps] = interp_SI_funs(I_in_t_block (:,[ 2 1]),I_in_t_block (:,3),x_range_tblck,[I_cur S_cur],0);

            xs_range_prt_elim(1).range = [0:.1:max(prt_elim_S(:,1))];
            xs_range_prt_elim(2).range = [0:.1:max(prt_elim_S(:,2))];
            %I, S
            [pred_prt_elim_1] = interp_SI_funs(prt_elim_S(:,[2 1]),prt_elim_S(:,3),xs_range_prt_elim,[I_cur min(130,S_cur)],0);

            %I, S
            [pred_prt_elim_2] = interp_SI_funs(prt_elim_S(:,[2 1]),prt_elim_S(:,4),xs_range_prt_elim,[I_cur min(130,S_cur)],0);

            prt_elim_perc= [pred_prt_elim_1 pred_prt_elim_2];
   
            %Pred facil window:
            
            full_elim_drop_1 = [250 250 250  250   50 12 1 250 250 250  250   50 12
                -60 -60 -60 -60 -60 -40 0     -60 -60 -60 -60 -60 -40 ];
            Ss = [132 85 55 30 14 5 0 130 86 56 29   12 7.05 ];
            x_range_S(1).range = [min(Ss):.1:max(Ss)];
            [z_drop_scale] = interp_SI_funs(Ss,full_elim_drop_1(1,:),x_range_S,S_cur,0);

            [dynl_scale] = interp_SI_funs(Ss,full_elim_drop_1(2,:),x_range_S,S_cur,0);

end
