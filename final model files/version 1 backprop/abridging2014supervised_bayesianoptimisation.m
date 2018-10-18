clear
clc
rng('default');


directory = 'D:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\data\abridging2014\nodirection\compression data';
X = csvread([directory filesep 'Abridging2014CompressionInputs.csv']); % input set
Y = csvread([directory filesep 'Abridging2014CompressionTargets.csv']); % target set
Xmean = csvread([directory filesep 'Abridging2014CompressionInputsMeans.csv']); % input set (means)
Ymean = csvread([directory filesep 'Abridging2014CompressionTargetsMeans.csv']); % target set (means)


% hyperparameter optimisation
optimVars = [
    optimizableVariable('NetworkDepth', [0 2], 'Type', 'integer') % test 1: [1 5], test 2: [0 2], test 3: [0 2]
    optimizableVariable('HiddenSize', [1 6], 'Type', 'integer') % test 1: [1 20], test 2: [1 6], test 3: [1 6]
    optimizableVariable('InitialLearnRate', [1e-2 0.4], 'Transform','log') % test 1: [1e-3 0.4], test 2: [1e-3 0.4], test 3: [1e-2, 0.4]
    optimizableVariable('Momentum', [0.6 0.95]) % test 1: [0.5 0.99], test 2: [0.8 0.99], test 3: [0.6 0.95] 
    % note: poslin = relu
    % optimizableVariable('ActivationFunction', {'logsig', 'tansig',
    % 'poslin'}, 'Type', 'categorical') % used in test 1 (all hidden)
    optimizableVariable('HiddenActivation1', {'logsig', 'tansig', 'poslin'}, 'Type', 'categorical') % used in test 2 & 3
    optimizableVariable('HiddenActivation2', {'logsig', 'tansig', 'poslin'}, 'Type', 'categorical') % used in test 2 & 3
    optimizableVariable('OutputActivation', {'logsig', 'tansig', 'poslin'}, 'Type', 'categorical')]; % used in test 2 & 3
    % optimizableVariable('TrainingFunction', {'traingd', 'traingdm', 'traingdx', 'trainrp', 'trainbr'}, 'Type', 'categorical') % test 1: ['traingd', 'traingdm', 'traingdx', 'trainrp', 'trainbr'], test 2: ['traingd', 'traingdm', 'trainbr'], test 3: just traingdm (comment out and set in objective function)
    % optimizableVariable('LossFunction', {'mse', 'sse', 'crossentropy'}, 'Type', 'categorical')]; % test 1: [mse, sse, crossentropy], test 2: sse, test 3: mse
optimResult = ffObjFcn(X, Y); 
BayesObjectTesticles = bayesopt(optimResult, optimVars, ...
    'MaxObj', 1000, ...
    'IsObjectiveDeterministic', false, ...
    'PlotFcn', {@plotMinObjective, @plotElapsedTime}); 

function optimResult = ffObjFcn(X, Y)
optimResult = @valErrorFun;
    function [valError, cons, fileName] = valErrorFun(optVars)
        if optVars.NetworkDepth == 0
            net = fitnet([]); 
        else 
            hiddenLayers = optVars.HiddenSize.*ones(1,optVars.NetworkDepth);
            net = fitnet(hiddenLayers);
            for i = 1:optVars.NetworkDepth
                % net.layers{i}.transferFcn = char(optVars.ActivationFunction); % for test 1
                if i == 1
                    net.layers{1}.transferFcn = char(optVars.HiddenActivation1);
                elseif i ==2 
                    net.layers{2}.transferFcn = char(optVars.HiddenActivation2); 
                end 
            end 
        end 
        net.layers{optVars.NetworkDepth + 1}.transferFcn = char(optVars.OutputActivation); % for tests 2 and 3
        
        % net.trainFcn = char(optVars.TrainingFunction); % for tests 1 & 2
        net.trainFcn = 'traingdm'; % for test 3

        net.trainParam.lr = optVars.InitialLearnRate;
        net.trainParam.time = 20; 
        if (strcmp(net.trainFcn, 'traingdm') || strcmp(net.TrainFcn, 'traingdx'))
            net.trainParam.mc = optVars.Momentum;
        end 
        % net.performFcn = char(optVars.LossFunction); % for test 1
        % net.performFcn = 'sse'; % for test 2
        net.performFcn = 'mse'; % for test 3
        
        [trainedNet, tr] = train(net, X', Y', 'useGPU', 'yes'); % NOTE: by default, feedforwardnets partition data 0.7:0.15:0.15 according to dividerand function
%         y = net(X'); 
%         perf = perform(net,y,X')
%         disp(tr)
        valError = tr.best_vperf;
        disp(tr)
        fileName = num2str(valError) + ".mat";
        dirFile = 'D:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\matlab\backpropOptim\test3\' + fileName; 
        save(dirFile, 'trainedNet', 'valError', 'net'); 
        cons = []; 
        
    end
end

