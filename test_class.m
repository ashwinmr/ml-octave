% Testing all functions in the class

clc;
clear all;
close all;

addpath('Functions');
addpath('Classes');

% Load training data
load('x');
load('y');

% Create an object of ML class
ml = ML_C();

% Set the x and y for the class
ml.x = x;
ml.y = y;

% Perform gradient descent
alpha = 0.01;
n = 1500;
ml.Gradient_Descent(alpha,n,1);

% Predict
y_predicted = ml.Predict(10);



%}