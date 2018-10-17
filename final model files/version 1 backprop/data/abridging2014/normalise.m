function normalised_data = normalise(data)
% Normalise values of an array to be between -1 and 1
% (original sign of array values is maintained)
if abs(min(data)) > max(data)
    max_range_value = abs(min(data));
    min_range_value = min(data);
else
    max_range_value = max(data);
    min_range_value = -max(data);
end

normalised_data = 2.*data./(max_range_value - min_range_value); 
end

