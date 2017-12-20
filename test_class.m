% Testing all functions in the class

clc;
clear;
close;

addpath('Functions');
addpath('Classes');

% Load training data
load('p');
load('v');

% Create an object of ML class
ml = ML_C();

% Set the x and y for the class
ml.x = p;
ml.theta = zeros(3,1);
ml.Set_y(v);

% Set theta

% Perform gradient descent
alpha = 0.01;
n = 10;
lambda = 0;
ml.Gradient_Descent_Custom(1,alpha,n,lambda);

% Predict
C = ml.theta(1);
K = ml.theta(2);
B = ml.theta(3);
P = ml.x;
V = (1-C./(P + C)).*(K.*(P+C)+B);
y_predicted = V;

% Plot
%{
plot(p,v,'*'); % Plot the training data
hold all;
plot(p(:,1),y_predicted); % Plot the prediction
grid on;
xlabel('x');
ylabel('y');
legend('Training Data','Prediction');
hold off;
%}