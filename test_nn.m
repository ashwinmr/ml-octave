% This script is used to test the neural network class

clc;
clear all;
close all;

% Add to path
addpath('Classes');
addpath('Functions');
addpath('Temp');

% Create neural network
nn = nn_c();

% Set properties
nn.nu = [400;25;10];

% Load Training Data
load('ex4data1.mat');

% Randomly select 100 data points to display
% sel = randperm(size(X, 1));
% sel = sel(1:100);

% displayData(X(sel, :));


% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');
% Unroll parameters 
nn_params = Unroll({Theta1;Theta2});

% Set theta
theta = Roll_Theta(nn_params,nn.nu);

% Compute cost and gradient
lambda = 3;
x = X; % x is same as training data
Y = y; % Store y in temporary variable
y = y==1:10; % y has to be for each class
[J,theta_grad] = nn.cost(x,y,theta,lambda);

% Do gradient checking using numerical gradient
% costFunc = @(p) nn.cost(x,y,Roll_Theta(p,nn.nu),lambda);
% numgrad = Num_Grad(costFunc,Unroll_Theta(nn.theta));

% Learning

max_iter = 50;
lambda = 1;
theta = nn.learn(x,y,max_iter,lambda);

% See what was learned
displayData(theta{1}(:,2:end));

% Prediction
pred = nn.predict(x);
[~,pred] = max(pred');
pred = pred(:);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);
