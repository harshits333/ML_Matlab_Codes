
%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 32;  % 20x20 Input Images of Digits
num_labels = 8;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('matlab.mat'); % training data stored in arrays x, y
m = size(x, 1);



fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 1;
[all_theta] = oneVsAll(x, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ setting x_train & y_train ================
xtrain=x;
ytrain=y;
load('matlabxy.mat');
xtest=x;
ytest=y;




%% ================ Plotting optimal lambda ================

%[lambda_vec, error_train, error_val] = ...
%    validationCurve(xtrain, ytrain, xtest, ytest,num_labels);
%
%close all;
%plot(lambda_vec, error_train, lambda_vec, error_val);
%legend('Train', 'Cross Validation');
%xlabel('lambda');
%ylabel('Accuracy');
%
%fprintf('lambda\t\tTrain Error\tValidation Error\n');
%for i = 1:length(lambda_vec)
%	fprintf(' %f\t%f\t%f\n', ...
%            lambda_vec(i), error_train(i), error_val(i));
%end
%
%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, xtrain);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytrain)) * 100);


pred = predictOneVsAll(all_theta, xtest);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

