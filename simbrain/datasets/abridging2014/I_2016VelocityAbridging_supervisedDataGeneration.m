%% Double brush 
%%%% CONSTANTS %%%%
clear 
count = 1; 
while (count < 101) 
    occluder_x1 = 5; occluder_x2 = 15; % cm
    occluder_width = occluder_x2 - occluder_x1; plane_width = 20; % cm
    A = 0; B = 0.04; C = 0.10; D = 0.16; E = 0.20; % target locations (m)
    brush_x1 = 0; brush_x2 = 10; % cm; CONSIDER CHANGING THIS TO 0.1 TO SIMPLIFY (only 0.09 in experiment to account for splaying / width of brush)
    delta_brush = brush_x2; % cm
    brushes = [brush_x1 brush_x2];
    time_steps_in = 25;
    time_steps_out = 10;
    delta_t_in = 0.030; % s
    delta_t_out = (25/10)*delta_t_in;
    % brushing_v = 0.15; % m/s
    
    %%%% VARIABLES %%%%
    velocity = (randperm(6,1) + 14); % cm/s
    brushing_v = velocity(1); % m/s; NOTE: 0.14 m/s is the slowest it can go to traverse the whole arm
    delta_x_in = brushing_v * delta_t_in; % m
    delta_x_out = brushing_v * delta_t_out; % m
    
    % traversal_time_apparent = (plane_width - brush_width)/brushing_v;
    % time_step = ceil(traversal_time_apparent / delta_t);
    %{
 So apparent time taken to traverse A-E or E-A = (20-9)/15 = 0.733 ~ 0.75s
 Let's say the total time for one test is 0.75s = 750ms
 So divide input space into 25 neurons, each successive neuron holding a
  value representing location of perceived stimulus @ time t
  where each successive adjacent neuron represents perceived location at
  certain time. Input layer: x(t) w/ delta(t) = 750/25 = 30ms
    %}
    
    
    for i = 1:time_steps_in
        if ((brushes(2) + delta_x_in) < plane_width)
            brushes = brushes + delta_x_in;
        else
            brushes(2) = plane_width;
        end
        
        if brushes(1) < occluder_x1
            input_row(i) = brushes(1);
        elseif (brushes(1) > occluder_x1) && (brushes(2) < occluder_x2)
            input_row(i) = -1;
            disp('Okay this is good')
        else
            input_row(i) = brushes(2);
        end
    end
    
    percept_tracker = 0;
    for j = 1:time_steps_out
        if ((percept_tracker + delta_x_out) < plane_width)
            percept_tracker = percept_tracker + delta_x_out;
        else
            percept_tracker = plane_width;
        end
        output_row(j) = percept_tracker;
    end
    
    input_data(count,:) = input_row; 
    output_data(count,:) = output_row;
    
    
    count = count + 1; 
end

% csvwrite('I_2016DoubleBrushSupervised_14to20cms_trainingOUT.csv', output_data)
% csvwrite('I_2016DoubleBrushSupervised_14to20cms_trainingIN.csv', input_data)
csvwrite('I_2016DoubleBrushSupervised_14to20cms_testingIN.csv', input_data)
%% Single brush

