clear
clc
rng('default');


directory = 'E:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\data\abridging2014\nodirection\compression data';
X = csvread([directory filesep 'Abridging2014CompressionInputs.csv']); % input set
Y = csvread([directory filesep 'Abridging2014CompressionTargets.csv']); % target set
Xmean = csvread([directory filesep 'Abridging2014CompressionInputsMeans.csv']); % input set (means)
Ymean = csvread([directory filesep 'Abridging2014CompressionTargetsMeans.csv']); % target set (means)

% NOTE: test 1 = generic, test 2 = for MATLAB modelling, test 3 = for Simbrain modelling
optimVars = [
    optimizableVariable('Depth', [0 3], 'Type', 'integer') % test 1b: [1 5], test 2b: [0 2], test 3b: [0 3], test 4: [0 3]
    optimizableVariable('Width', [1 10], 'Type', 'integer') % test 1b: [1 20], tests 2b & 3b: [1 15], test 4: [1 10]
    optimizableVariable('InitialLearnRate', [1e-3 0.5], 'Transform','log') % tests 1b, 3b, 4: [1e-3 0.5], test 2b: [1e-3 0.1]
    optimizableVariable('Momentum', [0.6 0.95]) % test 1b: [0.5 0.99], test 2b: 0.6:0.05:0.95, test 3b, 4: [0.6 0.95]
    % note: poslin = relu
    optimizableVariable('ActivationFunction', {'logsig', 'tansig','poslin','purelin'}, 'Type', 'categorical') % used in test 1 & 4 (all hidden layers)
%     optimizableVariable('HiddenActivation1', {'logsig', 'tansig', 'poslin', 'purelin'}, 'Type', 'categorical') % used in test 2 & 3 ~ test 2b: {'logsig', 'tansig', 'poslin'}, test 3b: {'logsig', 'tansig', 'poslin', 'purelin'}
%     optimizableVariable('HiddenActivation2', {'logsig', 'tansig', 'poslin', 'purelin'}, 'Type', 'categorical') % used in test 2 & 3 ~ test 2b: {'logsig', 'tansig', 'poslin'}, test 3b: {'logsig', 'tansig', 'poslin', 'purelin'}
%     optimizableVariable('HiddenActivation3', {'logsig', 'tansig', 'poslin', 'purelin'}, 'Type', 'categorical') % used in test 3b only
%     optimizableVariable('OutputActivation', {'purelin', 'tansig', 'poslin'}, 'Type', 'categorical') % used in test 2, 3, 4 
    optimizableVariable('TrainingFunction', {'traingdm', 'traingd', 'trainlm', 'trainbr', 'traingdx'}, 'Type', 'categorical')  % test 1b: {'traingd', 'traingdm', 'traingdx', 'trainrp', 'trainbr', 'trainlm'}, test 2b: {'traingdm', 'trainbr'}, test 3b: {'traingdm', 'traingd'}, test 4: every possible training function
    optimizableVariable('LossFunction', {'sse', 'mse'}, 'Type', 'categorical')]; % test 1b: {'mse', 'sse', 'mae'}, test 2b: {'sse'}, test 3: {'mse'}
optimResult = ffObjFcn(X, Y, 0); 
BayesObjectTest4 = bayesopt(optimResult, optimVars, ...
    'MaxObj', 1000, ...
    'IsObjectiveDeterministic', false, ...
    'PlotFcn', {@plotMinObjective, @plotElapsedTime, @plotObjectiveEvaluationTime, @plotObjective}, ...
    'UseParallel', false);  


function optimResult = ffObjFcn(X, Y, iter)
optimResult = @valErrorFun;
    function [valError, cons, fileName] = valErrorFun(optVars)
        iter = iter + 1; 
        if optVars.Depth == 0 % for tests 2, 3, 4
            net = fitnet([], char(optVars.TrainingFunction)); 
        else 
            hiddenLayers = optVars.Width.*ones(1,optVars.Depth);
            net = fitnet(hiddenLayers, char(optVars.TrainingFunction)); 
            for i = 1:optVars.Depth
                net.layers{i}.transferFcn = char(optVars.ActivationFunction); % for tests 1 & 4
                % if else clause below is for tests 2 & 3
%                 if i == 1
%                     net.layers{1}.transferFcn = char(optVars.HiddenActivation1);
%                 elseif i == 2 
%                     net.layers{2}.transferFcn = char(optVars.HiddenActivation2); 
%                 % final elseif is for test 3b only
%                 elseif i == 3
%                     net.layers{3}.transferFcn = char(optVars.HiddenActivation3); 
%                 end 
            end 
        end 
        % net.layers{optVars.Depth + 1}.transferFcn = char(optVars.OutputActivation); % for tests 2 & 3
        % net.trainFcn = char(optVars.TrainingFunction); % for tests 1, 2, 3, 4
        % net.trainFcn = 'trainbr'; % for test 4
        
        net.trainParam.lr = optVars.InitialLearnRate; 
        net.trainParam.time = 20; 
        if (strcmp(net.trainFcn, 'traingdm') || strcmp(net.TrainFcn, 'traingdx'))
            % two lines below are for tests 2b, 3b
            % momentum = (optVars.Momentum - mod(optVars.Momentum,5))/100; 
            % net.trainParam.mc = momentum; 
            
            net.trainParam.mc = optVars.Momentum; % for test 1, 2a, 3a, 4
        end 
        
        net.performFcn = char(optVars.LossFunction); 
%         net.performFcn = 'sse'; % for test 2b
%         net.performFcn = 'mse'; % for test 3b
                
        [trainedNet, tr] = train(net, X', Y'); % NOTE: by default, feedforwardnets partition data 0.7:0.15:0.15 according to dividerand function
%         y = net(X'); 
%         perf = perform(net,y,X')


        % valError = tr.best_vperf; % for any test not involving 'trainbr'
        valError = tr.best_perf; % for any test involving 'trainbr' (however, only results from test 4 used this; every other test has used best_vperf as objective)
        
        

        fileName = num2str(valError) + "_4_iter" + iter + ".mat";
        dirFile = 'E:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\matlab\backpropOptim\test4\4a\trainedNets\' + fileName; 
        save(dirFile, 'trainedNet', 'valError', 'net'); 
        cons = []; 
        
    end
end

