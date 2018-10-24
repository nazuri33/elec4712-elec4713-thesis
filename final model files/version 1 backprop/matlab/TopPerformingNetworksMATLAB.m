%% NET 1: 8 (1,8), tansig hidden, poslin output, trained with Bayesian regularization backprop, sse loss
clear
clc
rng('default');
directory = 'D:\checkout\elec4712-elec4713-thesis\final model files\version 1 backprop\data\abridging2014\nodirection\compression data';
X = csvread([directory filesep 'Abridging2014CompressionInputs.csv']); % input set
T = csvread([directory filesep 'Abridging2014CompressionTargets.csv']); % target set
Xmean = csvread([directory filesep 'Abridging2014CompressionInputsMeans.csv']); % input set (means)
Tmean = csvread([directory filesep 'Abridging2014CompressionTargetsMeans.csv']); % target set (means)
plot(X,T,'o')


net = fitnet(8, 'trainbr'); % tansig is default hidden layer, purelin is default output
net.performFcn = 'sse'; 
net = configure(net, X', T');
y1 = net(X'); 
[trainedNet, tr] = train(net,X',T','useParallel','yes', 'showResources', 'yes');
view(trainedNet)
y2 = trainedNet(X'); 
perf1 = perform(net,y1,T'); 
perf2 = perform(trainedNet,y2,T'); 
figure
plotperform(tr)
figure
plottrainstate(tr)
figure
plot(X',T', 'o', X', y1, 'x', X', y2, '*'); 




%% NET 2: 0 hidden, tansig output, trained with Bayesian regularization backprop, sse loss

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