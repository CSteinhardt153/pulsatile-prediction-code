function [pred_z] = interp_SI_funs(xs,y,x_range,xs_fit,plot_it)

if size(xs,1) < size(xs,2)
    xs = xs';
end
if size(y,1) < size(y,2)
    y = y';
end
if size(xs_fit,1) > size(xs_fit,2)
    xs_fit = xs_fit';
end

%Remove nans:
xs_clean = xs(find(~isnan(y)),:);
y_clean = y(find(~isnan(y)));

%1D
if size(xs_fit,2) == 1
    xq = x_range(1).range;
    pred_z = interp1(xs,y,xs_fit);
    if plot_it
        
        yq = interp1(xs,y,xq);
        [diff_1 pred_idx] = min(abs(xq - xs_fit));
        pred_z = yq(pred_idx);
        figure(1);
        plot(xs,y,'bo'); hold on;
        plot(xq,yq,'k');
        plot(xs_fit,pred_z,'r.','markersize',10);
        
        if (diff_1 > 5)
            warning('Prediction is out of interpolation boundaries');
        end
    end
    ylabel('Firing Rate (sps)'); xlabel('Pulse Rate (pps)')
else
    %2D
    x1_range = x_range(1).range;
    x2_range = x_range(2).range;
    
    pred_z = griddata(xs_clean(:,1),xs_clean(:,2),y_clean,xs_fit(2),xs_fit(1));
    [diff_1 closest_idx_1] = min(abs(x1_range - xs_fit(1)));
    [diff_2 closest_idx_2] = min(abs(x2_range - xs_fit(2)));
    if ((diff_1 - max(x1_range)) < 5)
        xs_fit(1) = x1_range(closest_idx_1);
    end
    if ((diff_2 - max(x2_range)) < 5)
        xs_fit(2) = x2_range(closest_idx_2);
    end
    pred_z = griddata(xs_clean(:,1),xs_clean(:,2),y_clean,xs_fit(2),xs_fit(1));
    
    if plot_it
        [diff_1 closest_idx_1] = min(abs(x1_range - xs_fit(1)));
        [diff_2 closest_idx_2] = min(abs(x2_range - xs_fit(2)));
        
        [xq,yq] = meshgrid(x2_range,x1_range); %reverses x,y axis
        zq = griddata(xs_clean(:,1),xs_clean(:,2),y_clean,xq,yq);
        %[x1_range(closest_idx_1),x2_range(closest_idx_2)]
        pred_z = zq(closest_idx_1,closest_idx_2);
        
        figure(2);
        mesh(yq,xq,zq); hold on;
        plot3(xs_clean(:,2),xs_clean(:,1),y_clean,'bo');
        plot3(xs_fit(1),xs_fit(2),pred_z,'r.','markersize',30)
        
        xlabel('SR')
        ylabel('I')
        zlabel('Function')
        
        if (diff_1 > 5)| (diff_2 > 5)
            warning('Prediction is out of interpolation boundaries');
            
        end
    end
end

if (isnan(pred_z))
    warning('Prediction is out of interpolation boundaries');
end
end