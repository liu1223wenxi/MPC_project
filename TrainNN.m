clear all;
close all;
clc;

%% Prepare Data
data = readtable('C:\Users\liu12\OneDrive - Florida Institute of Technology\Documents\Class\AEE5590\Code\MPC_project\State_Control.xlsx');

data.Gap = [];
data.time_u = [];
data = rmmissing(data);

% 80% train 20% validate
rng("default")
c = cvpartition(size(data,1),"Holdout",0.20);
trainingIdx = training(c); % Training set indices
dataTrain = data(trainingIdx,:);
testIdx = test(c); % Test set indices
dataTest = data(testIdx,:);

Mdl = fitrnet(dataTrain,"MPG","OptimizeHyperparameters","auto", ...
      "Verbose",1)


iteration = Mdl.TrainingHistory.Iteration;
trainLosses = Mdl.TrainingHistory.TrainingLoss;
valLosses = Mdl.TrainingHistory.ValidationLoss;
plot(iteration,trainLosses,iteration,valLosses)
legend(["Training","Validation"])
xlabel("Iteration")
ylabel("Mean Squared Error")

%%
% 
% c = cvpartition(size(data, 1), 'HoldOut', 0.2);
% idx = c.test;
% % Separate to training and test data
% trainData = data(~idx, :);
% testData = data(idx, :);
% 
% features = trainData{:, {'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'xdot', 'ydot', 'zdot', 'p', 'q', 'r'}};
% targets = trainData{:, {'thrust', 'rollAngle', 'pitchAngle', 'yawAngle'}};
% 
% regressionLearner
% 
% yfit = trainedModel.predictFcn(T)