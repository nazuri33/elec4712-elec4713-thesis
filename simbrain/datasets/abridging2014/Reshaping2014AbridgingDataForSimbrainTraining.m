%% Raw data --> structured table (see https://bit.ly/2O6PUe6)
array = table2array(Expt7AFilteredABCDEANALYSISTSCSARAHbaseline2);
array = reshape(array, [], 1);
array = array.';
sonya_baseline_2 = array2table(array); 

%% Creation of training sets from structured table
% double_brush_nodir_data = table2array(structuredTableDoubleBrushNoDir);
clear
Abridg2014ImportNoDirLocalisationData;
% annie = 2; fahed = 9; hamid = 16; julian = 23; nastaran = 30; norfizah = 37; paja = 44; rachel = 51; sarah = 58; sonya = 65; 
subject_rows = [2, 9, 16, 23, 30, 37, 44, 51, 58, 65];

% A. SHORT brushing
%   note: 24 short runs for each subject. Start w/ 12 for training, 12 for
%         testing (might need validation set, cross-validation later)

% we want responses 1-3 for training and 4-6 for testing

idx = 2; % responses 1-3 target columns (add 12 for responses 4-6)
idx2 = 1; % indexes to SHORT RUN row numbers
short_run_rows = [0, 1, 2, 5];

idx3 = 27; % mean baseline (inputs) columns
idx4 = 33; % responses 1-3 baseline (inputs) columns (add 12 for responses 4-6)
for m = 1:12
   if (idx == 14) idx = 2; end 
   if (idx4 == 48) idx4 = 33; end
   for n = 1:4
       annie_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(1)+short_run_rows(idx2), idx); annie_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(1)+short_run_rows(idx2), idx3); 
       annie_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(1)+short_run_rows(idx2), idx+12); 
       annie_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(1)+short_run_rows(idx2), idx4); 
       annie_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(1)+short_run_rows(idx2), idx4+15);
       
       fahed_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(2)+short_run_rows(idx2), idx); fahed_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(2)+short_run_rows(idx2), idx3); 
       fahed_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(2)+short_run_rows(idx2), idx+12); 
       fahed_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(2)+short_run_rows(idx2), idx4); 
       fahed_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(2)+short_run_rows(idx2), idx4+15);
       
       hamid_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(3)+short_run_rows(idx2), idx); hamid_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(3)+short_run_rows(idx2), idx3); 
       hamid_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(3)+short_run_rows(idx2), idx+12); 
       hamid_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(3)+short_run_rows(idx2), idx4); 
       hamid_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(3)+short_run_rows(idx2), idx4+15);
       
       julian_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(4)+short_run_rows(idx2), idx); julian_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(4)+short_run_rows(idx2), idx3); 
       julian_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(4)+short_run_rows(idx2), idx+12); 
       julian_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(4)+short_run_rows(idx2), idx4); 
       julian_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(4)+short_run_rows(idx2), idx4+15);
       
       nastaran_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(5)+short_run_rows(idx2), idx); nastaran_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(5)+short_run_rows(idx2), idx3); 
       nastaran_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(5)+short_run_rows(idx2), idx+12); 
       nastaran_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(5)+short_run_rows(idx2), idx4); 
       nastaran_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(5)+short_run_rows(idx2), idx4+15);
       
       norfizah_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(6)+short_run_rows(idx2), idx); norfizah_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(6)+short_run_rows(idx2), idx3); 
       norfizah_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(6)+short_run_rows(idx2), idx+12); 
       norfizah_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(6)+short_run_rows(idx2), idx4); 
       norfizah_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(6)+short_run_rows(idx2), idx4+15);
       
       paja_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(7)+short_run_rows(idx2), idx); paja_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(7)+short_run_rows(idx2), idx3); 
       paja_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(7)+short_run_rows(idx2), idx+12); 
       paja_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(7)+short_run_rows(idx2), idx4); 
       paja_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(7)+short_run_rows(idx2), idx4+15);
       
       rachel_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(8)+short_run_rows(idx2), idx); rachel_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(8)+short_run_rows(idx2), idx3); 
       rachel_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(8)+short_run_rows(idx2), idx+12); 
       rachel_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(8)+short_run_rows(idx2), idx4); 
       rachel_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(8)+short_run_rows(idx2), idx4+15);
       
       sarah_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(9)+short_run_rows(idx2), idx); sarah_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(9)+short_run_rows(idx2), idx3); 
       sarah_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(9)+short_run_rows(idx2), idx+12); 
       sarah_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(9)+short_run_rows(idx2), idx4); 
       sarah_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(9)+short_run_rows(idx2), idx4+15);
       
       sonya_training_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(10)+short_run_rows(idx2), idx); sonya_training_set_inputs(m,n) = abridging2014LocDataNoDir(subject_rows(10)+short_run_rows(idx2), idx3); 
       sonya_testing_set_targets(m, n) = abridging2014LocDataNoDir(subject_rows(10)+short_run_rows(idx2), idx+12); 
       sonya_training_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(10)+short_run_rows(idx2), idx4); 
       sonya_testing_set_inputs_1(m, n) = abridging2014LocDataNoDir(subject_rows(10)+short_run_rows(idx2), idx4+15);
       
       idx = idx + 1; 
       
       if (n == 2)
           idx3 = idx3 + 2;
           idx4 = idx4 + 2; 
       else
           idx3 = idx3 + 1;
           idx4 = idx4 + 1; 
       end
   end
   
   idx3 = 27; 
   if ((mod(m,3) == 0) && idx2 < 4) idx2 = idx2 + 1; end
end

%%% Setting table row/variable names and directories for exporting datasets
training_row_names = cellstr(["Short prelim (1)"; "Short prelim (2)"; "Short prelim (3)"; "Short 1 (1)"; "Short 1 (2)"; "Short 1 (3)"; "Short 2 (1)"; "Short 2 (2)"; "Short 2 (3)"; "Short 3 (1)"; "Short 3 (2)"; "Short 3 (3)"]);
testing_row_names = cellstr(["Short prelim (4)"; "Short prelim (5)"; "Short prelim (6)"; "Short 1 (4)"; "Short 1 (5)"; "Short 1 (6)"; "Short 2 (4)"; "Short 2 (5)"; "Short 2 (6)"; "Short 3 (4)"; "Short 3 (5)"; "Short 3 (6)"]);
training_row_input_names = cellstr(["Baseline prelim (1)"; "Baseline prelim (2)"; "Baseline prelim (3)"; "Baseline 1 (1)"; "Baseline 1 (2)"; "Baseline 1 (3)"; "B1(1)"; "B1(2)"; "B1(3)"; "Baseline 2 (1)"; "Baseline 2 (2)"; "Baseline 2 (3)"]);
testing_row_input_names = cellstr(["Baseline prelim (4)"; "Baseline prelim (5)"; "Baseline prelim (6)"; "Baseline 1 (4)"; "Baseline 1 (5)"; "Baseline 1 (6)"; "B1(4)"; "B1(5)"; "B1(6)"; "Baseline 2 (4)"; "Baseline 2 (5)"; "Baseline 2 (6)"]);
mean_input_row_names = cellstr(["Baseline prelim (mean)"; "bpm"; "."; "Baseline 1 (mean)"; "b1m"; ".."; "..."; "...."; "....."; "Baseline 2 (mean)"; "b2m"; "......"]);
var_names = {'A' 'B' 'D' 'E'}; 
training_target_data_directory = 'D:\Uniwork\6th year\Thesis\Milestones\10th of August milestones (SIMBRAIN)\Data\abridging2014\nodir\training\target'; training_input_data_directory = 'D:\Uniwork\6th year\Thesis\Milestones\10th of August milestones (SIMBRAIN)\Data\abridging2014\nodir\training\input';
testing_target_data_directory = 'D:\Uniwork\6th year\Thesis\Milestones\10th of August milestones (SIMBRAIN)\Data\abridging2014\nodir\testing\target'; testing_input_data_directory = 'D:\Uniwork\6th year\Thesis\Milestones\10th of August milestones (SIMBRAIN)\Data\abridging2014\nodir\testing\input';
validation_mean_input_data_directory = 'D:\Uniwork\6th year\Thesis\Milestones\10th of August milestones (SIMBRAIN)\Data\abridging2014\nodir\validation\input\means'; % validation_mean_target_data_directory = 'D:\Uniwork\6th year\Thesis\Milestones\10th of August milestones (SIMBRAIN)\Data\abridging2014\nodir\validation\target\means';

%%% Exporting CSVs %%%

% Annie
annie_training_set_inputs.Properties.RowNames = mean_input_row_names; annie_training_set_inputs.Properties.VariableNames = var_names; annie_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_ANNIE_inputs_nodir_2014abridging.csv']; writetable(annie_training_set_inputs, annie_training_inputs_csv); 
annie_training_set_targets.Properties.RowNames = training_row_names; annie_training_set_targets.Properties.VariableNames = var_names; annie_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_ANNIE_trainingtargets_nodir_2014abridging.csv']; writetable(annie_training_set_targets, annie_training_csv); 
annie_testing_set_targets.Properties.RowNames = testing_row_names; annie_testing_set_targets.Properties.VariableNames = var_names; annie_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_ANNIE_testingtargets_nodir_2014abridging.csv']; writetable(annie_testing_set_targets, annie_testing_csv);
annie_training_set_inputs_1.Properties.RowNames = training_row_input_names; annie_training_set_inputs_1.Properties.VariableNames = var_names; annie_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_ANNIE_traininginputs_nodir_2014abridging.csv']; writetable(annie_training_set_inputs_1, annie_training_inputs_1_csv); 
annie_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; annie_testing_set_inputs_1.Properties.VariableNames = var_names; annie_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_ANNIE_testinginputs_nodir_2014abridging.csv']; writetable(annie_testing_set_inputs_1, annie_testing_inputs_1_csv);
% Fahed
fahed_training_set_inputs.Properties.RowNames = mean_input_row_names; fahed_training_set_inputs.Properties.VariableNames = var_names; fahed_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_FAHED_inputs_nodir_2014abridging.csv']; writetable(fahed_training_set_inputs, fahed_training_inputs_csv); 
fahed_training_set_targets.Properties.RowNames = training_row_names; fahed_training_set_targets.Properties.VariableNames = var_names; fahed_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_FAHED_trainingtargets_nodir_2014abridging.csv']; writetable(fahed_training_set_targets, fahed_training_csv); 
fahed_testing_set_targets.Properties.RowNames = testing_row_names; fahed_testing_set_targets.Properties.VariableNames = var_names; fahed_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_FAHED_testingtargets_nodir_2014abridging.csv']; writetable(fahed_testing_set_targets, fahed_testing_csv);
fahed_training_set_inputs_1.Properties.RowNames = training_row_input_names; fahed_training_set_inputs_1.Properties.VariableNames = var_names; fahed_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_FAHED_traininginputs_nodir_2014abridging.csv']; writetable(fahed_training_set_inputs_1, fahed_training_inputs_1_csv); 
fahed_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; fahed_testing_set_inputs_1.Properties.VariableNames = var_names; fahed_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_FAHED_testinginputs_nodir_2014abridging.csv']; writetable(fahed_testing_set_inputs_1, fahed_testing_inputs_1_csv);
% Hamid
hamid_training_set_inputs.Properties.RowNames = mean_input_row_names; hamid_training_set_inputs.Properties.VariableNames = var_names; hamid_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_HAMID_inputs_nodir_2014abridging.csv']; writetable(hamid_training_set_inputs, hamid_training_inputs_csv); 
hamid_training_set_targets.Properties.RowNames = training_row_names; hamid_training_set_targets.Properties.VariableNames = var_names; hamid_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_HAMID_trainingtargets_nodir_2014abridging.csv']; writetable(hamid_training_set_targets, hamid_training_csv);
hamid_testing_set_targets.Properties.RowNames = testing_row_names; hamid_testing_set_targets.Properties.VariableNames = var_names; hamid_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_HAMID_testingtargets_nodir_2014abridging.csv']; writetable(hamid_testing_set_targets, hamid_testing_csv);
hamid_training_set_inputs_1.Properties.RowNames = training_row_input_names; hamid_training_set_inputs_1.Properties.VariableNames = var_names; hamid_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_HAMID_traininginputs_nodir_2014abridging.csv']; writetable(hamid_training_set_inputs_1, hamid_training_inputs_1_csv); 
hamid_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; hamid_testing_set_inputs_1.Properties.VariableNames = var_names; hamid_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_HAMID_testinginputs_nodir_2014abridging.csv']; writetable(hamid_testing_set_inputs_1, hamid_testing_inputs_1_csv);
% Julian
julian_training_set_inputs.Properties.RowNames = mean_input_row_names; julian_training_set_inputs.Properties.VariableNames = var_names; julian_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_JULIAN_inputs_nodir_2014abridging.csv']; writetable(julian_training_set_inputs, julian_training_inputs_csv); 
julian_training_set_targets.Properties.RowNames = training_row_names; julian_training_set_targets.Properties.VariableNames = var_names; julian_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_JULIAN_trainingtargets_nodir_2014abridging.csv']; writetable(julian_training_set_targets, julian_training_csv); 
julian_testing_set_targets.Properties.RowNames = testing_row_names; julian_testing_set_targets.Properties.VariableNames = var_names; julian_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_JULIAN_testingtargets_nodir_2014abridging.csv']; writetable(julian_testing_set_targets, julian_testing_csv);
julian_training_set_inputs_1.Properties.RowNames = training_row_input_names; julian_training_set_inputs_1.Properties.VariableNames = var_names; julian_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_JULIAN_traininginputs_nodir_2014abridging.csv']; writetable(julian_training_set_inputs_1, julian_training_inputs_1_csv); 
julian_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; julian_testing_set_inputs_1.Properties.VariableNames = var_names; julian_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_JULIAN_testinginputs_nodir_2014abridging.csv']; writetable(julian_testing_set_inputs_1, julian_testing_inputs_1_csv);
% Nastaran
nastaran_training_set_inputs.Properties.RowNames = mean_input_row_names; nastaran_training_set_inputs.Properties.VariableNames = var_names; nastaran_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_NASTARAN_inputs_nodir_2014abridging.csv']; writetable(nastaran_training_set_inputs, nastaran_training_inputs_csv); 
nastaran_training_set_targets.Properties.RowNames = training_row_names; nastaran_training_set_targets.Properties.VariableNames = var_names; nastaran_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_NASTARAN_trainingtargets_nodir_2014abridging.csv']; writetable(nastaran_training_set_targets, nastaran_training_csv);
nastaran_testing_set_targets.Properties.RowNames = testing_row_names; nastaran_testing_set_targets.Properties.VariableNames = var_names; nastaran_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_NASTARAN_testingtargets_nodir_2014abridging.csv']; writetable(nastaran_testing_set_targets, nastaran_testing_csv);
nastaran_training_set_inputs_1.Properties.RowNames = training_row_input_names; nastaran_training_set_inputs_1.Properties.VariableNames = var_names; nastaran_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_NASTARAN_traininginputs_nodir_2014abridging.csv']; writetable(nastaran_training_set_inputs_1, nastaran_training_inputs_1_csv); 
nastaran_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; nastaran_testing_set_inputs_1.Properties.VariableNames = var_names; nastaran_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_NASTARAN_testinginputs_nodir_2014abridging.csv']; writetable(nastaran_testing_set_inputs_1, nastaran_testing_inputs_1_csv);
% Norfizah
norfizah_training_set_inputs.Properties.RowNames = mean_input_row_names; norfizah_training_set_inputs.Properties.VariableNames = var_names; norfizah_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_NORFIZAH_inputs_nodir_2014abridging.csv']; writetable(norfizah_training_set_inputs, norfizah_training_inputs_csv); 
norfizah_training_set_targets.Properties.RowNames = training_row_names; norfizah_training_set_targets.Properties.VariableNames = var_names; norfizah_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_NORFIZAH_trainingtargets_nodir_2014abridging.csv']; writetable(norfizah_training_set_targets, norfizah_training_csv); 
norfizah_testing_set_targets.Properties.RowNames = testing_row_names; norfizah_testing_set_targets.Properties.VariableNames = var_names; norfizah_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_NORFIZAH_testingtargets_nodir_2014abridging.csv']; writetable(norfizah_testing_set_targets, norfizah_testing_csv);
norfizah_training_set_inputs_1.Properties.RowNames = training_row_input_names; norfizah_training_set_inputs_1.Properties.VariableNames = var_names; norfizah_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_NORFIZAH_traininginputs_nodir_2014abridging.csv']; writetable(norfizah_training_set_inputs_1, norfizah_training_inputs_1_csv); 
norfizah_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; norfizah_testing_set_inputs_1.Properties.VariableNames = var_names; norfizah_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_NORFIZAH_testinginputs_nodir_2014abridging.csv']; writetable(norfizah_testing_set_inputs_1, norfizah_testing_inputs_1_csv);
% Paja
paja_training_set_inputs.Properties.RowNames = mean_input_row_names; paja_training_set_inputs.Properties.VariableNames = var_names; paja_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_PAJA_inputs_nodir_2014abridging.csv']; writetable(paja_training_set_inputs, paja_training_inputs_csv); 
paja_training_set_targets.Properties.RowNames = training_row_names; paja_training_set_targets.Properties.VariableNames = var_names; paja_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_PAJA_trainingtargets_nodir_2014abridging.csv']; writetable(paja_training_set_targets, paja_training_csv); 
paja_testing_set_targets.Properties.RowNames = testing_row_names; paja_testing_set_targets.Properties.VariableNames = var_names; paja_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_PAJA_testingtargets_nodir_2014abridging.csv']; writetable(paja_testing_set_targets, paja_testing_csv);
paja_training_set_inputs_1.Properties.RowNames = training_row_input_names; paja_training_set_inputs_1.Properties.VariableNames = var_names; paja_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_PAJA_traininginputs_nodir_2014abridging.csv']; writetable(paja_training_set_inputs_1, paja_training_inputs_1_csv); 
paja_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; paja_testing_set_inputs_1.Properties.VariableNames = var_names; paja_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_PAJA_testinginputs_nodir_2014abridging.csv']; writetable(paja_testing_set_inputs_1, paja_testing_inputs_1_csv);
% Rachel
rachel_training_set_inputs.Properties.RowNames = mean_input_row_names; rachel_training_set_inputs.Properties.VariableNames = var_names; rachel_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_RACHEL_inputs_nodir_2014abridging.csv']; writetable(rachel_training_set_inputs, rachel_training_inputs_csv); 
rachel_training_set_targets.Properties.RowNames = training_row_names; rachel_training_set_targets.Properties.VariableNames = var_names; rachel_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_RACHEL_trainingtargets_nodir_2014abridging.csv']; writetable(rachel_training_set_targets, rachel_training_csv); 
rachel_testing_set_targets.Properties.RowNames = testing_row_names; rachel_testing_set_targets.Properties.VariableNames = var_names; rachel_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_RACHEL_testingtargets_nodir_2014abridging.csv']; writetable(rachel_testing_set_targets, rachel_testing_csv);
rachel_training_set_inputs_1.Properties.RowNames = training_row_input_names; rachel_training_set_inputs_1.Properties.VariableNames = var_names; rachel_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_RACHEL_traininginputs_nodir_2014abridging.csv']; writetable(rachel_training_set_inputs_1, rachel_training_inputs_1_csv); 
rachel_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; rachel_testing_set_inputs_1.Properties.VariableNames = var_names; rachel_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_RACHEL_testinginputs_nodir_2014abridging.csv']; writetable(rachel_testing_set_inputs_1, rachel_testing_inputs_1_csv);
% Sarah
sarah_training_set_inputs.Properties.RowNames = mean_input_row_names; sarah_training_set_inputs.Properties.VariableNames = var_names; sarah_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_SARAH_inputs_nodir_2014abridging.csv']; writetable(sarah_training_set_inputs, sarah_training_inputs_csv); 
sarah_training_set_targets.Properties.RowNames = training_row_names; sarah_training_set_targets.Properties.VariableNames = var_names; sarah_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_SARAH_trainingtargets_nodir_2014abridging.csv']; writetable(sarah_training_set_targets, sarah_training_csv); 
sarah_testing_set_targets.Properties.RowNames = testing_row_names; sarah_testing_set_targets.Properties.VariableNames = var_names; sarah_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_SARAH_testingtargets_nodir_2014abridging.csv']; writetable(sarah_testing_set_targets, sarah_testing_csv);
sarah_training_set_inputs_1.Properties.RowNames = training_row_input_names; sarah_training_set_inputs_1.Properties.VariableNames = var_names; sarah_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_SARAH_traininginputs_nodir_2014abridging.csv']; writetable(sarah_training_set_inputs_1, sarah_training_inputs_1_csv); 
sarah_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; sarah_testing_set_inputs_1.Properties.VariableNames = var_names; sarah_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_SARAH_testinginputs_nodir_2014abridging.csv']; writetable(sarah_testing_set_inputs_1, sarah_testing_inputs_1_csv);
% Sonya
sonya_training_set_inputs.Properties.RowNames = mean_input_row_names; sonya_training_set_inputs.Properties.VariableNames = var_names; sonya_training_inputs_csv = [validation_mean_input_data_directory filesep 'shortonly_allmeans_combo1_bpb1b1b2_SONYA_inputs_nodir_2014abridging.csv']; writetable(sonya_training_set_inputs, sonya_training_inputs_csv); 
sonya_training_set_targets.Properties.RowNames = training_row_names; sonya_training_set_targets.Properties.VariableNames = var_names; sonya_training_csv = [training_target_data_directory filesep 'shortonly_combo1_5050_resps1to3_SONYA_trainingtargets_nodir_2014abridging.csv']; writetable(sonya_training_set_targets, sonya_training_csv); 
sonya_testing_set_targets.Properties.RowNames = testing_row_names; sonya_testing_set_targets.Properties.VariableNames = var_names; sonya_testing_csv = [testing_target_data_directory filesep 'shortonly_combo1_5050_resps4to6_SONYA_testingtargets_nodir_2014abridging.csv']; writetable(sonya_testing_set_targets, sonya_testing_csv);
sonya_training_set_inputs_1.Properties.RowNames = training_row_input_names; sonya_training_set_inputs_1.Properties.VariableNames = var_names; sonya_training_inputs_1_csv = [training_input_data_directory filesep 'shortonly_combo1_5050_resps1to3_SONYA_traininginputs_nodir_2014abridging.csv']; writetable(sonya_training_set_inputs_1, sonya_training_inputs_1_csv); 
sonya_testing_set_inputs_1.Properties.RowNames = testing_row_input_names; sonya_testing_set_inputs_1.Properties.VariableNames = var_names; sonya_testing_inputs_1_csv = [testing_input_data_directory filesep 'shortonly_combo1_5050_resps4to6_SONYA_testinginputs_nodir_2014abridging.csv']; writetable(sonya_testing_set_inputs_1, sonya_testing_inputs_1_csv);






    