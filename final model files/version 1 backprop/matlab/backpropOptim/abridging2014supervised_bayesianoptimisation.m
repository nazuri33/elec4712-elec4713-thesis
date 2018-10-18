clear
clc
rng('default');


directory = 'E:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\data\abridging2014\nodirection\compression data';
X = csvread([directory filesep 'Abridging2014CompressionInputs.csv']); % input set
Y = csvread([directory filesep 'Abridging2014CompressionTargets.csv']); % target set
Xmean = csvread([directory filesep 'Abridging2014CompressionInputsMeans.csv']); % input set (means)
Ymean = csvread([directory filesep 'Abridging2014CompressionTargetsMeans.csv']); % target set (means)

% NOTE: test 1 = generic, test 2 = for Python modelling, test 3 = for Simbrain modelling
optimVars = [
    optimizableVariable('Depth', [0 3], 'Type', 'integer') % test 1b: [1 5], test 2b: [0 2], test 3b: [0 3]
    optimizableVariable('Width', [1 15], 'Type', 'integer') % test 1b: [1 20], tests 2b & 3b: [1 15]
    optimizableVariable('InitialLearnRate', [1e-3 0.5], 'Transform','log') % tests 1b & 3b: [1e-3 0.5], test 2b: [1e-3 0.1]
    optimizableVariable('Momentum', [60 95], 'Type', 'integer') % test 1b: [0.5 0.99], test 2b: 0.6:0.05:0.95, test 3b: [0.6 0.95] 
    % note: poslin = relu
    % optimizableVariable('ActivationFunction', {'logsig', 'tansig','poslin','purelin'}, 'Type', 'categorical') % used in test 1 (all layers)
    optimizableVariable('HiddenActivation1', {'logsig', 'tansig', 'poslin', 'purelin'}, 'Type', 'categorical') % used in test 2 & 3 ~ test 2b: {'logsig', 'tansig', 'poslin'}, test 3b: {'logsig', 'tansig', 'poslin', 'purelin'}
    optimizableVariable('HiddenActivation2', {'logsig', 'tansig', 'poslin', 'purelin'}, 'Type', 'categorical') % used in test 2 & 3 ~ test 2b: {'logsig', 'tansig', 'poslin'}, test 3b: {'logsig', 'tansig', 'poslin', 'purelin'}
    optimizableVariable('HiddenActivation3', {'logsig', 'tansig', 'poslin', 'purelin'}, 'Type', 'categorical') % used in test 3b only
    optimizableVariable('OutputActivation', {'purelin', 'tansig', 'poslin'}, 'Type', 'categorical') % used in test 2 and 3 (which both use same values for test b)
    optimizableVariable('TrainingFunction', {'traingdm', 'traingd'}, 'Type', 'categorical')];  % test 1b: {'traingd', 'traingdm', 'traingdx', 'trainrp', 'trainbr', 'trainlm'}, test 2b: {'traingdm', 'trainbr'}, test 3b: {'traingdm', 'traingd'}
    % optimizableVariable('LossFunction', {'sse', 'mse', 'mae'}, 'Type',
    % 'categorical')]; % test 1b: {'mse', 'sse', 'mae'}, test 2b: {'sse'}, test 3: {'mse'}
optimResult = ffObjFcn(X, Y, 0); 
BayesObjectTest3b = bayesopt(optimResult, optimVars, ...
    'MaxObj', 1000, ...
    'IsObjectiveDeterministic', false, ...
    'PlotFcn', {@plotMinObjective, @plotElapsedTime}, ...
    'UseParallel', true); 

function optimResult = ffObjFcn(X, Y, iter)
optimResult = @valErrorFun;
    function [valError, cons, fileName] = valErrorFun(optVars)
        iter = iter + 1; 
        if optVars.Depth == 0 % for tests 2 and 3
            net = fitnet([]); 
        else 
            hiddenLayers = optVars.Width.*ones(1,optVars.Depth);
            net = fitnet(hiddenLayers);
            for i = 1:optVars.Depth
%                 net.layers{i}.transferFcn = char(optVars.ActivationFunction); % for test 1
                if i == 1
                    net.layers{1}.transferFcn = char(optVars.HiddenActivation1);
                elseif i == 2 
                    net.layers{2}.transferFcn = char(optVars.HiddenActivation2); 
                % final elseif is for test 3b only
                elseif i == 3
                    net.layers{3}.transferFcn = char(optVars.HiddenActivation3); 
                end 
            end 
        end 
        net.layers{optVars.Depth + 1}.transferFcn = char(optVars.OutputActivation); % for tests 2 and 3
        net.trainFcn = char(optVars.TrainingFunction); 

        net.trainParam.lr = optVars.InitialLearnRate;
        net.trainParam.time = 20; 
        if (strcmp(net.trainFcn, 'traingdm') || strcmp(net.TrainFcn, 'traingdx'))
%             net.trainParam.mc = optVars.Momentum;
            momentum = (optVars.Momentum - mod(optVars.Momentum,5))/100;  % for tests 2b and 3b
            net.trainParam.mc = momentum; 
          
        end 
%         net.performFcn = char(optVars.LossFunction); % for test 1b
%         net.performFcn = 'sse'; % for test 2b
        net.performFcn = 'mse'; % for test 3b
        
        [trainedNet, tr] = train(net, X', Y', 'useGPU', 'yes', 'showResources', 'yes'); % NOTE: by default, feedforwardnets partition data 0.7:0.15:0.15 according to dividerand function
%         y = net(X'); 
%         perf = perform(net,y,X')
%         disp(tr)
        valError = tr.best_vperf;
%         disp(tr)
        iter = iter + 1; 
        fileName = num2str(valError) + "_3b_iter" + iter + ".mat";
        dirFile = 'E:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\matlab\backpropOptim\test3\3b\trainedNets\' + fileName; 
        save(dirFile, 'trainedNet', 'valError', 'net'); 
        cons = []; 
        
    end
end

