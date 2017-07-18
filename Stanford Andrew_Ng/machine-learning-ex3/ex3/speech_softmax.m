%==========Softmax Classifier For Speech Recognition ========================


clear ; close all; clc


inputSize = 32; % Size of input vector (MNIST images are 28x28)
numClasses = 8;     % Number of classes (MNIST images fall into 10 classes)

%lambda = 1e-4; % Weight decay parameter
lambda = 0.1;
%%======================================================================
%% STEP 1: Load data

load('matlab.mat');
%m = size(x, 1);
x=x';
y=y';
%inputData = images;



%%======================================================================
%% STEP 4: Learning parameters
%
%  Once you have verified that your gradients are correct, 
%  you can start training your softmax regression code using softmaxTrain
%  (which uses minFunc).

options.maxIter = 1000;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            x, y, options);
                          
% Although we only use 100 iterations here to train a classifier for the 
% MNIST data set, in practice, training for more iterations is usually
% beneficial.

%%======================================================================
%% STEP 5: Testing
xtrain=x;
ytrain=y;
[pred] = softmaxPredict(softmaxModel, xtrain);

acc = mean(ytrain(:) == pred(:));
fprintf('Train Accuracy: %0.3f%%\n', acc * 100);


load('matlabxy.mat');
xtest=x';
ytest=y';
[pred] = softmaxPredict(softmaxModel, xtest);

acc = mean(ytest(:) == pred(:));
fprintf('Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% After 100 iterations, the results for our implementation were:
%
% Accuracy: 92.200%
%
% If your values are too low (accuracy less than 0.91), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
