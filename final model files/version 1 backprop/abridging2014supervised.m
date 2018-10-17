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
    optimizableVariable('NetworkDepth', [1 5], 'Type', 'integer')
    optimizableVariable('HiddenSize', [1 20], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-3 0.4], 'Transform','log')
    optimizableVariable('Momentum', [0.5 0.99])
    optimizableVariable('ActivationFunction', {'logsig', 'tansig', 'poslin'}, 'Type', 'categorical') % note: poslin is the same as relu
    optimizableVariable('TrainingFunction', {'traingd', 'traingdm', 'traingdx', 'trainrp', 'trainbr'}, 'Type', 'categorical') 
    optimizableVariable('LossFunction', {'mse', 'sse', 'crossentropy'}, 'Type', 'categorical')]; 
optimResult = ffObjFcn(X, Y); 
BayesObject = bayesopt(optimResult, optimVars, ...
    'MaxObj', 1000, ...
    'IsObjectiveDeterministic', false, ...
    'UseParallel', true); 



% % cross-validation w/ final network
% Y_part = cvpartition(Y, 'KFold', 5); 
% for i = 1:Y_part.NumTestSets
%     % train, validate, test?
%     disp(['Fold ', num2str(i)])
% %     testClasses = Y(Y_part.test(i));
% %     testClasses
%     trIdx = Y_part.training(i);
%     testIdx = Y_part.test(i); 
%     trainCnt = sum(trIdx == 1);
%     testCnt = sum(testIdx == 1);
%     disp('Train:test ratio: ')
%     trainCnt/(trainCnt + testCnt)
%     
% %     disp('Training input');
%     XTrain = X(trIdx, :);
% %     disp('Testing input');
%     XTest = X(testIdx, :);
% %     disp('Training target');
%     YTrain = Y(trIdx);
% %     disp('Testing target'); 
%     YTest = Y(testIdx);
%     
%     valIdx = randperm(numel(YTrain), ceil(0.2*numel(YTrain))); 
%     XValidation = XTrain(valIdx, :);
%     XTrain(valIdx, :) = [];
%     YValidation = YTrain(valIdx); 
%     YTrain(valIdx, :) = []; 
%     
% end 

% mse = crossval('mse',X,Y,'Predfun',traingdm); 

function optimResult = ffObjFcn(X, Y)
optimResult = @valErrorFun;
    function [valError, cons, fileName] = valErrorFun(optVars)
        hiddenLayers = optVars.HiddenSize.*ones(1,optVars.NetworkDepth);
        net = fitnet(hiddenLayers);
        for i = 1:optVars.NetworkDepth
%             disp(optVars.ActivationFunction)
            net.layers{i}.transferFcn = char(optVars.ActivationFunction); 
        end 
        
        net.trainFcn = char(optVars.TrainingFunction);
        net.trainParam.lr = optVars.InitialLearnRate;
%         net.trainParam.goal = 0.001;
%         net.trainParam.min_grad = 1e-4; 
        net.trainParam.time = 20; 
        if (strcmp(optVars.TrainingFunction, 'traingdm') || strcmp(optVars.TrainingFunction, 'traingdx'))
            net.trainParam.mc = optVars.Momentum;
        end 
        net.performFcn = char(optVars.LossFunction);
        
        [trainedNet, tr] = train(net, X', Y'); % NOTE: by default, feedforwardnets partition data 0.7:0.15:0.15 according to dividerand function
%         y = net(X'); 
%         perf = perform(net,y,X')
%         disp(tr)
        valError = tr.best_vperf;
        fileName = num2str(valError) + ".mat";
        dirFile = 'D:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\matlab\backpropOptim' + fileName; 
        save(dirFile, 'trainedNet', 'valError', 'net'); 
        cons = []; 
        
    end
end



function testval = train_net (XTRAIN, YTRAIN, XTEST, YTEST)

    net = feedforwardnet(10);
    [net, tr] = train(net, XTRAIN', YTRAIN');
    plotperform(tr); 
    
    % all of the below is for classification tasks 
    yNet = net(XTEST');
    %'// find which output (of the three dummy variables) has the highest probability
    [~,classNet] = max(yNet',[],2);

    %// convert YTEST into a format that can be compared with classNet
    [~,classTest] = find(YTEST);


    %'// Check the success of the classifier
    cp = classperf(classTest, classNet);
    testval = cp.CorrectRate; %// replace this with your preferred metric
    
end 