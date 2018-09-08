%% Save tables as csvs
% Note: before running this script, need to open all subjects *_byresponse.mat' files
directory = 'E:\checkout\elec4712-elec4713-thesis\simbrain\datasets\abridging2014\nodirection\reduced supervised model data\double'; 
subjects = {'sonya', 'annie', 'fahed', 'hamid', 'julian', 'nastaran', 'norfizah', 'paja', 'rachel', 'sarah'}; 

for i = 1:length(subjects)
    varnames = who(strcat('*_', subjects{i}));  
    for j = 1:12
        file_name = strcat(varnames(j), '.csv'); 
%       file_dir = [directory filesep file_name];
        current_table = table2array(varnames(j)); 
%       writetable(current_table, current_file); 
%       disp(varnames(n));
%       disp(current_table(1,1)); 
%         v = genvarname(current_table, who);
%         eval(v ' = current_table']);
        eval(['dataset = ',current_table,';']);
        writetable(dataset, file_name{1}); 
      
    end 
end

%% Construct data 1: index matrix for baseline (input) -> response (target) mapping
single_short = [0.7, 2]; % i.e. 700ms traverse time, 2 up-and-down brush sweeps
double_short = [0.1, 2]; % i.e. 100ms, 2 up-and-down sweeps
single_long = [0.7, 10]; % i.e. 700ms, 10 up-and-down sweeps
double_long = [0.1, 10]; % i.e. 100ms, 10 up-and-down sweeps
% We want every possible combination of inputs to targets.
    % Therefore, we want 6^2 permutations (6C2) of two-element vectors
    % MATLAB offers nchoosek function but not pchoosek. 
    % Since we're dealing with 2 column matrix, we can simply flip combntn
    %   matrix horizontally to get perm. matrix.
    %   We also want repeats so append these @ end then randomise row order
    % Resultant matrix is used as as indices to map inputs -> targets for
    %   thorough cross-validation
responses = 6; % 6 runs for each subject
variables = 2; % traverse time, exposure to brushing
combos = nchoosek(1:responses, variables); 
perm_count = length(combos)*factorial(variables); 
perm_count_wrepeats = responses*variables; 
idx = 1; 
% NOTE: this method will only work w/ 2 column (variable) matrix
for m = 1:perm_count_wrepeats
%     perms(m,:) = randperm(6,2); 
   if m > perm_count
       perms(m,:) = m - perm_count; 
   else 
       perms(m,:) = combos(idx, :); 
   end 
   
   if (m == perm_count/variables)
       combos = fliplr(combos);
       idx = 1; 
   else
       idx = idx + 1;
   end 
end

% shuffle row order
perms = perms(randperm(size(perms,1)),:); 

%% Construct data 2: creating dataset
subjects = {'sonya', 'annie', 'fahed', 'hamid', 'julian', 'nastaran', 'norfizah', 'paja', 'rachel', 'sarah'}; 
short_runs = 4; 
target_locations = 4;
perms_per_pair = (short_runs^2)*target_locations; 

for x = 1:length(subjects)
    tab_name = strcat(
    target_vars = who(strcat('response*', subjects{i}));
    input_vars = who(strcat('base*', subjects{i}));
    for x = 1:perm_count_wrepeats
%         index_in = perms(x,1);
%         index_targ = perms(x,2); 
        in_name = table2array(target_vars(perms(x,1))); 
        targ_name = table2array(target_vars(perms(x,2))); 
        eval(['current_in = ',in_name,';']);
        eval(['current_targ = ',targ_name,';']);

        for y = 
        end
    end 
end 








