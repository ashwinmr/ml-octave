% Testing all functions

%clc;
%clear all;
%close all;

addpath('Functions');
addpath('Classes');

% Load training data
load('x');
load('y');



% Add bias to x
x = Add_Bias(x);

% Feature normalize x
[x,mu,sigma] = Feature_Normalize(x);

% Perform gradient descent
alpha = 0.01;
n = 1500;
theta = zeros(size(x,2),1);
theta = Gradient_Descent(x,y,theta,alpha,n,1);