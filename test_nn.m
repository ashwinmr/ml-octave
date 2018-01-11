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

% Init theta
% nn.init_theta();

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');
% Unroll parameters 
nn_params = Unroll_Theta({Theta1;Theta2});

% Set theta
nn.theta = Roll_Theta(nn_params,nn.nu);

% Compute cost
lambda = 0;
x = X; % x is same as training data
y = y==1:10; % y has to be for each class
[J,theta_grad] = nn.cost(x,y,nn.theta,lambda);
