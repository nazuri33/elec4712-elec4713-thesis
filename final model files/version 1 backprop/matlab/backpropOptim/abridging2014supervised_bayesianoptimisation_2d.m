%% dataset initialisation
clear
clc
rng('default');


directory = 'E:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\data\abridging2014\nodirection\compression data';
X = csvread([directory filesep 'Abridging2014CompressionInputs.csv']); % input set
Y = csvread([directory filesep 'Abridging2014CompressionTargets.csv']); % target set
Xmean = csvread([directory filesep 'Abridging2014CompressionInputsMeans.csv']); % input set (means)
Ymean = csvread([directory filesep 'Abridging2014CompressionTargetsMeans.csv']); % target set (means)
%% test 5
optimVars5 = [
    optimizableVariable('Depth', [0 3], 'Type', 'integer') % for test 5
    optimizableVariable('Width', [1 10], 'Type', 'integer')]; % for test 5
optimResultTest5 = ffObjFcn5(X, Y, 0); 
BayesObjectTest5 = bayesopt(optimResultTest5, optimVars5, ...
    'MaxObj', 1000, ...
    'IsObjectiveDeterministic', false, ...
    'PlotFcn', {@plotAcquisitionFunction, @plotConstraintModels, @plotObjectiveEvaluationTimeModel, @plotObjectiveModel, @plotObjective, @plotObjectiveEvaluationTime, @plotMinObjective, @plotElapsedTime}, ...
    'UseParallel', true); 
%% test 6
optimVars6 = [
    optimizableVariable('InitialLearnRate', [1e-3 0.5], 'Transform','log') % for test 6
    optimizableVariable('Momentum', [0.6 0.95])]; % for test 6
    % note: poslin = relu
optimResultTest6 = ffObjFcn6(X, Y, 0); 
BayesObjectTest6 = bayesopt(optimResultTest6, optimVars6, ...
    'MaxObj', 1000, ...
    'IsObjectiveDeterministic', false, ...
    'PlotFcn', {@plotAcquisitionFunction, @plotConstraintModels, @plotObjectiveEvaluationTimeModel, @plotObjectiveModel, @plotObjective, @plotObjectiveEvaluationTime, @plotMinObjective, @plotElapsedTime}, ...
    'UseParallel', true); 

%  x = bestPoint(BayesObjectTest4, 'Criterion', 'min-observed'); 

%% objective functions
% test 5
function optimResultTest5 = ffObjFcn5(X5, Y5, iter5)
optimResultTest5 = @valErrorFun5;
    function [valError5, cons5, fileName5] = valErrorFun5(optVars5)
        iter5 = iter5 + 1; 
        if optVars5.Depth == 0 
            net5 = fitnet([], 'traingdm'); 
        else 
            hiddenLayers = optVars5.Width.*ones(1,optVars5.Depth);
            net5 = fitnet(hiddenLayers, 'traingdm'); 
            for i = 1:optVars5.Depth
                net5.layers{i}.transferFcn = 'poslin';
            end 
        end 
            
        net5.performFcn = 'mse'; 
        net5.trainParam.lr = 0.25;
        net5.trainparam.mc = 0.9;
        [trainedNet5, tr5] = train(net5, X5', Y5', 'useGPU', 'yes'); % NOTE: by default, feedforwardnets partition data 0.7:0.15:0.15 according to dividerand function

        valError5 = tr5.best_perf; 
        

        fileName5 = num2str(valError5) + "_5b_iter" + iter5 + ".mat";
        dirFile5 = 'E:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\matlab\backpropOptim\test5\5b\trainedNets\' + fileName5; 
        save(dirFile5, 'trainedNet5', 'valError5', 'net5'); 
        cons5 = []; 
    end
end

% test 6
function optimResultTest6 = ffObjFcn6(X6, Y6, iter6)
optimResultTest6 = @valErrorFun6;
    function [valError6, cons6, fileName6] = valErrorFun6(optVars6)
        iter6 = iter6 + 1; 
        net6 = fitnet(3, 'traingdm'); 
        net6.layers{1}.transferFcn = 'poslin'; 
        net6.performFcn = 'mse';
       
        net6.trainParam.lr = optVars6.InitialLearnRate; 
        net6.trainparam.mc = optVars6.Momentum; 
        [trainedNet6, tr6] = train(net6, X6', Y6', 'useGPU', 'yes'); % NOTE: by default, feedforwardnets partition data 0.7:0.15:0.15 according to dividerand function

        valError6 = tr6.best_perf; 
        

        fileName6 = num2str(valError6) + "_6b_iter" + iter6 + ".mat";
        dirFile6 = 'E:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\matlab\backpropOptim\test6\6b\trainedNets\' + fileName6; 
        save(dirFile6, 'trainedNet6', 'valError6', 'net6'); 
        cons6 = []; 

    end
end





